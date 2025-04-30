import torch
import os
import logging
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from vlmv2 import VLM, VLMConfig, BirdVQADataset, MyDataCollator
from tqdm import tqdm  # for progress bar


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_vlm(
    llm_model_path="microsoft/phi-3-mini-4k-instruct",
    vision_model_path="tinyvit_cub200.pth",
    vision_projection_type="simple",
    image_root="/workspace/Treehouse/CUB_200_2011/images",
    json_path="train_qa.json",
    
    # training param
    output_dir="vlm_output_final",
    batch_size=4,
    learning_rate=1e-5,
    num_epochs=2,
    warmup_percent=0.1,
    gradient_accumulation_steps=1,
    gradient_clip=0.5,
    use_combined_loss=True,
    alignment_loss_weight=0.1,
    num_workers=8,
    device="cuda"
):
     
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    logger.info("Creating model...")
    vlm_config = VLMConfig(
        llm_model_path=llm_model_path,
        vision_model_path=vision_model_path,
        use_combined_loss=use_combined_loss,
        alignment_loss_weight=alignment_loss_weight,
        vision_projection_type=vision_projection_type,
    )
    model = VLM(vlm_config)
    model.to(device)
    
    logger.info("Loading dataset...")
    dataset = BirdVQADataset(
        image_root=image_root,
        json_path=json_path,
        tokenizer=model.tokenizer,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=MyDataCollator(model.tokenizer),
        num_workers=num_workers,
        drop_last=False
    )
    
    # set up optimizer - only train projection layers
    logger.info("Setting up optimizer...")
    trainable_params = []
    for name, param in model.named_parameters():
        if any(substr in name for substr in ["linear", "text_pooler", "image_pooler"]):
            param.requires_grad = True
            trainable_params.append(param)
            logger.info(f"Training parameter: {name}, shape={param.shape}")
        else:
            param.requires_grad = False
    
    # create optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # calculate total steps
    total_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
    
  
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_percent * total_steps),
        num_training_steps=total_steps
    )
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    global_step = 0

    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0)
    
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0.0
    
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                          leave=False, position=1, 
                          total=len(dataloader))
        
        for batch in batch_pbar:
            #to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            #forward pass
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
           
            batch_pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})
            
           
            if (batch_pbar.n + 1) % gradient_accumulation_steps == 0 or (batch_pbar.n + 1) == len(dataloader):
                # clip gardient
                torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            
                global_step += 1
            
            
            epoch_loss += loss.item() * gradient_accumulation_steps
        
      
        avg_loss = epoch_loss / len(dataloader)
        epoch_pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})
        logger.info(f"Epoch {epoch+1} complete, Average Loss: {avg_loss:.4f}")
        
        
        checkpoint_dir = f"{output_dir}/checkpoint_epoch_{epoch+1}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # only sace the trainable layers 
        state_dict = model.state_dict()
        for key in list(state_dict.keys()):
            if any(substr in key for substr in ["linear", "text_pooler", "image_pooler"]):
                torch.save({key: state_dict[key]}, f"{checkpoint_dir}/{key.replace('.', '_')}.pt")
    
   
    epoch_pbar.close()
    
    logger.info("Saving final model...")
    final_dir = f"{output_dir}/final_model"
    os.makedirs(final_dir, exist_ok=True)
    
    state_dict = model.state_dict()
    for key in list(state_dict.keys()):
        if any(substr in key for substr in ["linear", "text_pooler", "image_pooler"]):
            torch.save({key: state_dict[key]}, f"{final_dir}/{key.replace('.', '_')}.pt")
    
    model.tokenizer.save_pretrained(f"{output_dir}/tokenizer")
    logger.info(f"Training complete. Model saved to {output_dir}")
    
    return model

if __name__ == "__main__":
    torch.cuda.empty_cache()
    train_vlm()
    