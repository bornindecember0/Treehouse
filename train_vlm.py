import torch
import os
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from vlm import VLM, VLMConfig, BirdVQADataset, MyDataCollator

# configuration
config = {
    "llm_model_path": "microsoft/phi-3-mini-4k-instruct",
    "vision_model_path": "tinyvit_cub200.pth",
    "image_root": "cub_split/train", 
    "json_path": "QnA_datapreprocess/train_qa.json", 
    "output_dir": "phi3_vlm_output",
    "batch_size": 4,
    "learning_rate": 5e-5,
    "num_epochs": 3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "validate_samples": False
}

def save_checkpoint(model, epoch, step, output_dir):
    """Save model state dict only - with minimal disk usage"""
    import shutil
    
    # delete previous checkpoints to save space
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if item.startswith("checkpoint_") and os.path.isdir(item_path):
            print(f"Removing previous checkpoint: {item_path}")
            shutil.rmtree(item_path)
    
    # create new checkpoint directory
    if step > 0:
        checkpoint_dir = f"{output_dir}/checkpoint_step_{step}"
    else:
        checkpoint_dir = f"{output_dir}/checkpoint_epoch_{epoch}"
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Get state dict
    state_dict = model.state_dict()
    
    # save each layer separately
    for key in list(state_dict.keys()):
        if "linear" in key:  # only save linear layer weights!!!!
            try:
                torch.save({key: state_dict[key]}, f"{checkpoint_dir}/{key.replace('.', '_')}.pt")
            except Exception as e:
                print(f"Error saving {key}: {e}")
    
    print(f"Checkpoint saved to {checkpoint_dir}")
    return checkpoint_dir

def train():
    print("Creating model...")
    vlm_config = VLMConfig(
        llm_model_path=config["llm_model_path"],
        vision_model_path=config["vision_model_path"]
    )
    model = VLM(vlm_config)
    model.to(config["device"])
    
    print("Loading dataset...")
    dataset = BirdVQADataset(
        image_root=config["image_root"],
        json_path=config["json_path"],
        tokenizer=model.tokenizer,
        skip_validation=True  
    )
    
    print(f"Starting with dataset of {len(dataset)} samples")
    collator = MyDataCollator(model.tokenizer)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        drop_last=False
    )
    
    # optimizer - only train the projection layers!!!
    trainable_params = [p for n, p in model.named_parameters() if "linear" in n]
    print(f"Training {len(trainable_params)} parameters")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config["learning_rate"]
    )
    
  
    total_steps = len(dataloader) * config["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"Starting training for {config['num_epochs']} epochs...")
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        valid_batch_count = 0
        
        for step, batch in enumerate(dataloader):
            try:
                batch = {k: v.to(config["device"]) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                valid_batch_count += 1
                
                if step % 10 == 0:
                    print(f"Epoch {epoch+1}/{config['num_epochs']}, Step {step}, Loss: {loss.item():.4f}")
                
            except Exception as e:
                print(f"Error in training batch {step}: {e}")
                optimizer.zero_grad()
                continue
        
        # save epoch checkpoint - only at the end of the epoch
        if valid_batch_count > 0:
            avg_loss = total_loss / valid_batch_count
            print(f"Epoch {epoch+1} complete, Average Loss: {avg_loss:.4f}")
            
            # save model checkpoint at end of epoch only
            save_checkpoint(model, epoch+1, 0, config["output_dir"])
        else:
            print(f"Epoch {epoch+1} had no valid batches, skipping checkpoint")
    
  
    model.tokenizer.save_pretrained(f"{output_dir}/tokenizer")
    print(f"Training complete. Model saved to {config['output_dir']}")

    #new! call evaluation
    evaluate_vlm(model, train_dataset, tokenizer)

if __name__ == "__main__":
    os.makedirs(config["output_dir"], exist_ok=True)
    train()
