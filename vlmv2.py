import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
import os
from typing import List, Dict, Any
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VLMConfig(PretrainedConfig):
    model_type = "vlm_model"
    def __init__(self,
                 llm_model_path="microsoft/phi-3-mini-4k-instruct",
                 vision_model_path="tinyvit_cub200.pth",
                 freeze_vision_model=True,
                 image_pad_num=49,
                 use_combined_loss=False,  # disabled by default for initial stability
                 alignment_loss_weight=0.1,
                 vision_projection_type="simple",  # 'simple' or 'complex'
                 **kwargs):
        self.llm_model_path = llm_model_path
        self.vision_model_path = vision_model_path
        self.freeze_vision_model = freeze_vision_model
        self.image_pad_num = image_pad_num
        self.use_combined_loss = use_combined_loss
        self.alignment_loss_weight = alignment_loss_weight
        self.vision_projection_type = vision_projection_type
        super().__init__(**kwargs)

class VLM(PreTrainedModel):
    config_class = VLMConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config 
        #vision model
        self.vision_model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0) #num classes = 0 here !
        state_dict = torch.load(config.vision_model_path, map_location="cpu")
        # remove classification head weights
        state_dict = {k: v for k, v in state_dict.items() if "head" not in k}
        self.vision_model.load_state_dict(state_dict, strict=False)

        #LLM and tokenizer
        self.llm_model = AutoModelForCausalLM.from_pretrained(config.llm_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)

        # add <|image_pad|> token 
        if "<|image_pad|>" not in self.tokenizer.get_vocab():
            logger.info("Adding <|image_pad|> token to vocabulary")
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['<|image_pad|>']})
            self.llm_model.resize_token_embeddings(len(self.tokenizer))
            
        # Ensure pad token is set (important for Phi-3)
        if self.tokenizer.pad_token is None:
            logger.info("Setting pad_token to eos_token")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        vision_dim = self.vision_model.num_features  # 192 for tiny_vit
        llm_dim = self.llm_model.config.hidden_size

        # projection type based on config
        if config.vision_projection_type == "simple":
            logger.info("using simple linear projection for vision features")
            self.linear_projection = nn.Sequential(
                nn.Linear(vision_dim, llm_dim),
                nn.LayerNorm(llm_dim),
            )
        else:
            logger.info("using complex projection with bottleneck for vision features")
            self.linear_projection = nn.Sequential(
                nn.Linear(vision_dim, llm_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(llm_dim // 2, llm_dim),
                nn.LayerNorm(llm_dim)
            )
         
        # for alignment loss 
        if config.use_combined_loss:
            logger.info("Initializing poolers for combined loss")
            # text representation extractor
            self.text_pooler = nn.Sequential(
                nn.Linear(llm_dim, llm_dim),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim // 2)
            )
            
            # image representation pooler
            self.image_pooler = nn.Sequential(
                nn.Linear(llm_dim, llm_dim),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim // 2)
            )
    
        # freeze vision model and LLM weights
        if config.freeze_vision_model:
            logger.info("Freezing vision model parameters")
            for p in self.vision_model.parameters():
                p.requires_grad = False
                
        logger.info("Freezing LLM model parameters")
        for p in self.llm_model.parameters():
            p.requires_grad = False

        # image preprocessing transform
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image):
        return self.image_transform(image)

    def forward(self, input_ids, labels, pixel_values, attention_mask=None, **kwargs,):
        device = input_ids.device
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        vision_embeds = self.vision_model.forward_features(pixel_values)  # [B, 197, 192]
        
        vision_feature_size = vision_embeds.shape[1] - 1  # - CLS token
        num_tokens = min(self.config.image_pad_num, vision_feature_size)
        vision_tokens = vision_embeds[:, 1:num_tokens+1, :]  # skip CLS token
       
        # project vision features to LLM dimension
        image_features = self.linear_projection(vision_tokens)  # [B, rokens, llm_dim]
        
        text_embeds = text_embeds.to(image_features.dtype)
        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)


        # forward pass through LLM
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask,
            output_hidden_states=self.config.use_combined_loss  # only need hidden states for alignment loss
        )
        logits = outputs.logits
        
       
        loss = None
        self.lm_loss = None
        self.alignment_loss = None
        
        if labels is not None:
            #llm loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            self.lm_loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1).to(logits.device)
            )
            
            # just llm loss
            loss = self.lm_loss.clone()
            
            # alignment loss 
            if self.config.use_combined_loss and hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                self.alignment_loss = self._calculate_alignment_loss(
                    outputs.hidden_states[-1], 
                    labels, 
                    image_features
                )
                # combine losses
                loss = self.lm_loss + self.config.alignment_loss_weight * self.alignment_loss
                if torch.rand(1).item() < 0.01:  # 1% chance to print
                    logger.info(f"LM Loss: {self.lm_loss.item():.4f}, Alignment Loss: {self.alignment_loss.item():.4f}")

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    # for alignement loss        
    def _calculate_alignment_loss(self, hidden_states, labels, image_features):
        # extract valid text representations (non -100 in labels)
        valid_mask = (labels != -100).float().unsqueeze(-1)
        masked_hidden = hidden_states * valid_mask
        text_sum = masked_hidden.sum(dim=1)
        text_count = valid_mask.sum(dim=1) + 1e-8  # avoid div by zero
        text_vec = text_sum / text_count
        
        # process through text pooler
        text_vec = F.dropout(text_vec, p=0.2, training=self.training)
        text_vec = self.text_pooler(text_vec)
        
        # pool image features
        image_vec = image_features.mean(dim=1)
        image_vec = F.dropout(image_vec, p=0.2, training=self.training)
        image_vec = self.image_pooler(image_vec)
        
        # normalize vectors for cosine similarity
        text_vec = F.normalize(text_vec, p=2, dim=1)
        image_vec = F.normalize(image_vec, p=2, dim=1)
        
        # handle batch size cases
        batch_size = text_vec.size(0)
        
        if batch_size <= 1:
            # direct similarity loss 
            alignment_loss = 1.0 - F.cosine_similarity(text_vec, image_vec).mean()
            alignment_loss = alignment_loss + 0.1  # add min threshold
        else:
            # compute similarity matrix with temperature
            temperature = 0.1
            similarity = torch.matmul(text_vec, image_vec.transpose(0, 1)) / temperature
            
            # label smoothing
            smooth_factor = 0.1
            target_dist = torch.ones_like(similarity) * smooth_factor / (batch_size - 1)
            target_dist.fill_diagonal_(1.0 - smooth_factor)
            
            # alignment loss
            alignment_loss = (
                -torch.sum(target_dist * F.log_softmax(similarity, dim=1)) / batch_size
                -torch.sum(target_dist * F.log_softmax(similarity.t(), dim=1)) / batch_size
            ) / 2.0
            
            # add minimum loss threshold to prevent collapse
            alignment_loss = alignment_loss + 0.1
            
        return alignment_loss



    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
        try:
            pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            new_embeds = inputs_embeds.clone()
            
            # Calculate norm ratio for proper scaling
            text_norm = torch.norm(inputs_embeds, dim=2).mean().item()
            image_norm = torch.norm(image_features, dim=2).mean().item()
            
            # Scale image features to be comparable to text (slightly stronger)
            scale_factor = (text_norm / image_norm) * 5  
            
            # Apply scaling
            scaled_image_features = image_features * scale_factor
            
            # Log the norms for debugging
            scaled_norm = torch.norm(scaled_image_features, dim=2).mean().item()
            logger.info(f"Norms - Text: {text_norm:.4f}, Image (original): {image_norm:.4f}, Image (scaled): {scaled_norm:.4f}")
            # TEXT: ~ 0.8  IMG (scaled) ~1.2 / IMG(ori) ~55
            for b in range(input_ids.shape[0]):
                pad_positions = (input_ids[b] == pad_id).nonzero(as_tuple=False)
                if pad_positions.size(0) == 0:
                    logger.warning(f"No image pad tokens found in batch {b}")
                    continue
                if pad_positions.size(1) > 1:
                    pad_positions = pad_positions.squeeze(-1)
                else:
                    pad_positions = pad_positions.view(-1)
                num_features = min(scaled_image_features.size(1), pad_positions.size(0))
                
                for i in range(num_features):
                    new_embeds[b, pad_positions[i]] = scaled_image_features[b, i]
                    
            return new_embeds
        except Exception as e:
            logger.warning(f"merge_input_ids_with_image_features failed: {e}")
            return inputs_embeds
  
    
    def generate_with_vision_features(self, input_ids, attention_mask, pixel_values, max_new_tokens=50, **kwargs):
        device = input_ids.device
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        
        # Add debugging for input prompt
        logger.info(f"Generation prompt: {self.tokenizer.decode(input_ids[0])[:100]}...")
        
        # Check for image tokens
        pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        pad_count = (input_ids[0] == pad_id).sum().item()
        logger.info(f"Image pad tokens found: {pad_count}")
        
        # Extract vision features with debugging
        vision_embeds = self.vision_model.forward_features(pixel_values)
        vision_feature_size = vision_embeds.shape[1] - 1  # Minus CLS token
        num_tokens = min(self.config.image_pad_num, vision_feature_size)
        vision_tokens = vision_embeds[:, 1:num_tokens+1, :]
        image_features = self.linear_projection(vision_tokens)
        
        # Debug feature statistics
        logger.info(f"Vision features - mean: {image_features.mean().item():.4f}, std: {image_features.std().item():.4f}")
        logger.info(f"Text embeddings - mean: {text_embeds.mean().item():.4f}, std: {text_embeds.std().item():.4f}")
        
        # Ensure same dtype
        text_embeds = text_embeds.to(image_features.dtype)
        
        # Apply strong scaling to make image features more prominent
        # This is a temporary test to see if scaling helps
        # image_features = image_features * 2.0  # Increase visual feature scale as a test
        
        # Merge image features into text embeddings
        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
        
        # Debug merged embeddings
        logger.info(f"Merged embeddings - mean: {inputs_embeds.mean().item():.4f}, std: {inputs_embeds.std().item():.4f}")
        
        # Check if image features were actually inserted
        text_only_positions = [(input_ids[0] != pad_id).nonzero(as_tuple=True)[0].cpu().tolist()]
        pad_positions = [(input_ids[0] == pad_id).nonzero(as_tuple=True)[0].cpu().tolist()]
        text_embeds_norm = torch.norm(inputs_embeds[0, text_only_positions], dim=1).mean().item()
        image_embeds_norm = torch.norm(inputs_embeds[0, pad_positions], dim=1).mean().item() if pad_positions[0] else 0
        logger.info(f"Text positions norm: {text_embeds_norm}, Image positions norm: {image_embeds_norm}")
        
        generation_kwargs = {
            "do_sample": True,  
            "temperature": 0.2, 
            "top_p": 0.9, 
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": max_new_tokens
        }
        
        torch.cuda.empty_cache()
                
        generated_ids = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs
        )
        
        # Debug generated output
        if generated_ids.size(0) > 0:
            decoded = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
            logger.info(f"Generated (with special tokens): {decoded[:100]}...")
        
        return generated_ids


    
class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, features):
        if not features:
            return None
            
        # if in evaluation mode 
        is_eval_mode = 'labels' not in features[0]
        
        input_ids = [f['input_ids'] for f in features]
        max_len = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        
        for ids in input_ids:
            padding = [self.tokenizer.pad_token_id] * (max_len - len(ids))
            padded_input_ids.append(ids + padding)
            

        attention_mask = []
        for ids in input_ids:
            mask = [1] * len(ids) + [0] * (max_len - len(ids))
            attention_mask.append(mask)
            
        # in training mode
        if not is_eval_mode:
            labels = [f['labels'] for f in features]
            padded_labels = []
            
            for l in labels:
                label_pad_len = max_len - len(l)
                padded_labels.append(l + [-100] * label_pad_len)
                
        
        pixel_values = [f['pixel_values'] for f in features]
        
        batch = {
            'input_ids': torch.tensor(padded_input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'pixel_values': torch.stack(pixel_values)
        }
        
        if not is_eval_mode:
            batch['labels'] = torch.tensor(padded_labels)
            
        return batch        
   

class BirdVQADataset(Dataset):
    def __init__(self, image_root, json_path, tokenizer, image_pad_num=49):
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.pad_token = "<|image_pad|>"
        self.image_pad_num = image_pad_num
        if self.pad_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({'additional_special_tokens': [self.pad_token]})
            logger.info(f"Added {self.pad_token} token to vocabulary")
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token")

        #image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

        with open(json_path, 'r') as f:
            self.data = json.load(f)
        logger.info(f"Loaded {len(self.data)} samples from JSON file")
        
        # build label to folder mapping
        self.label_to_folder = {}
        self._build_label_folder_mapping()
        
    def _build_label_folder_mapping(self):
        # path structure: /workspace/Treehouse/CUB_200_2011/images/001.Black_footed_Albatross/file_name
        
        if not os.path.exists(self.image_root):
            logger.warning(f"Image root directory not found: {self.image_root}")
            return
            
        for folder in os.listdir(self.image_root):
            if '.' in folder:
                # extract label number ("001" from "001.Black_footed_Albatross")
                label_str = folder.split('.')[0]
                label = int(label_str)
                self.label_to_folder[label] = {
                    'folder': folder,
                    'path': os.path.join(self.image_root, folder)
                }
             
        
        logger.info(f"Found {len(self.label_to_folder)} label-to-folder mappings")

    def get_image_path(self, filename, label):
        # Check if we have a mapping for this label
        if label not in self.label_to_folder:
            logger.warning(f"No folder mapping found for label {label}")
            return None
            
        # Construct the full path
        folder_info = self.label_to_folder[label]
        return os.path.join(folder_info['path'], filename)

    def __len__(self):
        return len(self.data)

       
    def evaluation_mode(self, enable=True):
        self.eval_mode = enable

    def __getitem__(self, idx):
        sample = self.data[idx]
        filename = sample['file_name']
        label = sample['label']
        image_path = self.get_image_path(filename, label)
        
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            
        img = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(img)
        
        question = sample['question']
        if isinstance(sample['answer'], list):
            answer = ", ".join(sample['answer'])
        else:
            answer = str(sample['answer'])
            
        chat = [
            {"role": "system", "content": "You are a helpful bird expert. Answer questions about birds based on the image provided. Keep answers brief and direct."},
            {"role": "user", "content": question + self.pad_token * self.image_pad_num}
        ]
        
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)
        
        if hasattr(self, 'eval_mode') and self.eval_mode:
            input_ids = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
            return {
                'input_ids': input_ids,
                'pixel_values': pixel_values
            }
        else:
            # training - include answer
            full_chat = chat + [{"role": "assistant", "content": answer}]
            full_text = self.tokenizer.apply_chat_template(full_chat, tokenize=False)
            answer_text = full_text[len(prompt):]
            q_ids = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
            a_ids = self.tokenizer(answer_text, add_special_tokens=False)['input_ids']
            
            input_ids = q_ids + a_ids
            labels = [-100] * len(q_ids) + a_ids
            
            return {
                'input_ids': input_ids,
                'labels': labels,
                'pixel_values': pixel_values
            }