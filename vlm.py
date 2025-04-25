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

class VLMConfig(PretrainedConfig):
    model_type = "vlm_model"
    def __init__(self,
                 llm_model_path="microsoft/phi-3-mini-4k-instruct",  # Phi-3
                 vision_model_path="tinyvit_cub200.pth",
                 freeze_vision_model=True,
                 image_pad_num=49,
                 **kwargs):
        self.llm_model_path = llm_model_path
        self.vision_model_path = vision_model_path
        self.freeze_vision_model = freeze_vision_model
        self.image_pad_num = image_pad_num
        super().__init__(**kwargs)

class VLM(PreTrainedModel):
    config_class = VLMConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        #vision model
        self.vision_model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
        state_dict = torch.load(config.vision_model_path, map_location="cpu")
        # remove classification head weights
        state_dict = {k: v for k, v in state_dict.items() if "head" not in k}
        self.vision_model.load_state_dict(state_dict, strict=False)

        #LLM and tokenizer
        self.llm_model = AutoModelForCausalLM.from_pretrained(config.llm_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)

        # add <|image_pad|> 
        if "<|image_pad|>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['<|image_pad|>']})
            self.llm_model.resize_token_embeddings(len(self.tokenizer))
            
        # ensure pad token is set (important for Phi-3)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        vision_dim = self.vision_model.num_features #192
        llm_dim = self.llm_model.config.hidden_size

        # # v1 -> projection layers
        # self.linear1 = nn.Linear(197 * vision_dim, llm_dim)  # 37824 -> llm_dim
        # self.linear2 = nn.Linear(llm_dim, llm_dim) 

        #  v2 -> projection layers for multiple tokens
        self.visual_projection = nn.Sequential(
            nn.Linear(vision_dim, llm_dim // 2),  # Bottleneck to prevent overfitting
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(llm_dim // 2, llm_dim),
            nn.LayerNorm(llm_dim)
        )
         
    
        # freeze vit and llm weights
        if config.freeze_vision_model:
            for p in self.vision_model.parameters():
                p.requires_grad = False
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

    def forward(self, input_ids, labels, pixel_values, attention_mask=None):
        
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        vision_embeds = self.vision_model.forward_features(pixel_values)  # ([B, 197, 192])

        vision_tokens = vision_embeds[:, 1:self.config.image_pad_num+1, :]  # [B, 49, 192] # skip cls
        
        image_features = self.visual_projection(vision_tokens)  # [B, 49, llm_dim]
        text_embeds = text_embeds.to(image_features.dtype)
        
        # Merge image features into token embeddings
        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)

        # Forward pass
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs.logits
         
        # v1 
        # b, n, d = vision_embeds.shape
        # vision_embeds_flat = vision_embeds.reshape(b, n * d)  # ([B, 37824])

        # # project to match LLM dimensions
        # image_features = self.linear2(F.silu(self.linear1(vision_embeds_flat)))
       
        # text_embeds = text_embeds.to(image_features.dtype)

        # # merge image features into token embeddings
        # inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)

        # # forward 
        # outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        # logits = outputs.logits

        # compute loss 
        loss = None
        if labels is not None:
            # changed to use -100 as ignore_index for Phi-3
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1).to(logits.device)
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
        # add a check here 
        try:
            pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            new_embeds = inputs_embeds.clone()
            
            for b in range(input_ids.shape[0]):
                # Find positions of all pad tokens for this batch item
                pad_positions = (input_ids[b] == pad_id).nonzero(as_tuple=False).squeeze(-1)
                
                if pad_positions.numel() == 0:
                    continue
                
                # Insert image features at consecutive pad positions
                num_features = min(image_features.size(1), pad_positions.numel())
                for i in range(num_features):
                    new_embeds[b, pad_positions[i]] = image_features[b, i]
                
            return new_embeds
                    
        except Exception as e:
            print(f"[Warning] merge_input_ids_with_image_features failed: {e}")
            return inputs_embeds
 
    # v1
    # def generate_with_vision_features(self, input_ids, attention_mask, vision_features, max_new_tokens=100, **kwargs):
    #     """Generate text using the LLM with vision features."""
    #     # Get text embeddings from input IDs
    #     text_embeds = self.llm_model.get_input_embeddings()(input_ids)
    #     text_embeds = text_embeds.to(vision_features.dtype)
        
    #     # Merge image features into text embeddings
    #     inputs_embeds = self.merge_input_ids_with_image_features(vision_features, text_embeds, input_ids)
        
    #     # Generate text
    #     generated_ids = self.llm_model.generate(
    #         inputs_embeds=inputs_embeds,
    #         attention_mask=attention_mask,
    #         max_new_tokens=max_new_tokens,
    #         **kwargs
    #     )
        
    #     return generated_ids
    # v2
    def generate_with_vision_features(self, input_ids, attention_mask, pixel_values, max_new_tokens=100, **kwargs):
        # Get text embeddings
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        
        # Get vision features and project them
        vision_embeds = self.vision_model.forward_features(pixel_values)
        vision_tokens = vision_embeds[:, 1:self.config.image_pad_num+1, :]  # [B, 49, 192]
        image_features = self.visual_projection(vision_tokens)  # [B, 49, llm_dim]
        
        # Ensure same dtype
        text_embeds = text_embeds.to(image_features.dtype)
        
        # Merge image features into text embeddings
        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
        
        # Generate text
        generated_ids = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
        return generated_ids

def find_image_path(image_root, file_name):
    for dirpath, _, filenames in os.walk(image_root):
        if file_name in filenames:
            return os.path.join(dirpath, file_name)
    return None


class BirdVQADataset(Dataset):
    def __init__(self, image_root, json_path, tokenizer, image_pad_num=49, skip_validation=False):
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.pad_token = "<|image_pad|>"
        self.image_pad_num = image_pad_num
        self.skip_validation = skip_validation

        # add image_pad token if not already in vocabulary
        if self.pad_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({'additional_special_tokens': [self.pad_token]})
            print(f"Added {self.pad_token} token to vocabulary")
            
        # ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Set pad_token to eos_token")


        with open(json_path, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} samples from JSON")

        # Build image map recursively for efficient lookup
        self.image_map = {}
        print(f"Building image map from {image_root}...")
        
        # Expand search to include other directories
        search_paths = [image_root]
        parent_dir = os.path.dirname(image_root)
        for subdir in ['train', 'test', 'val', 'validation']:
            potential_path = os.path.join(parent_dir, subdir)
            if os.path.exists(potential_path) and potential_path != image_root:
                search_paths.append(potential_path)
                print(f"Adding additional search path: {potential_path}")
        
        # Search for images in all paths
        for search_path in search_paths:
            if os.path.exists(search_path):
                for root, _, files in os.walk(search_path):
                    for fn in files:
                        if fn.lower().endswith((".jpg")):
                            self.image_map[fn] = os.path.join(root, fn)
        
        print(f"Found {len(self.image_map)} images in directory tree")
        
        # skip validation ?
        if not skip_validation:
            # Quick validation - just check a sample of images
            sample_size = min(100, len(self.data))
            sample_indices = random.sample(range(len(self.data)), sample_size)
            missing = 0
            for idx in sample_indices:
                fn = self.data[idx]['file_name']
                if fn not in self.image_map:
                    missing += 1
            print(f"Validation sample: {missing}/{sample_size} images missing")
        
       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            sample = self.data[idx]
            fn = sample['file_name']
            image_path = self.image_map.get(fn)
            if not image_path:
                raise FileNotFoundError(f"Image '{fn}' not found in any search path")
            img = Image.open(image_path).convert("RGB")
            # pixel_values = self.transform(img)  # Shape: [3, 224, 224]
            question = sample['question']
            if isinstance(sample['answer'], list):
                answer = ", ".join(sample['answer'])
            else:
                answer = str(sample['answer'])

            # Create chat template
            chat = [
                {"role": "system", "content": "You are a helpful bird expert."},
                {"role": "user", "content": question + self.pad_token * self.image_pad_num}
            ]
            
            prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)
            full_chat = chat + [{"role": "assistant", "content": answer}]
            full_text = self.tokenizer.apply_chat_template(full_chat, tokenize=False)
            
            # the assistant part for labels (everything after the prompt)
            answer_text = full_text[len(prompt):]
            
            q_ids = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
            a_ids = self.tokenizer(answer_text, add_special_tokens=False)['input_ids']

            if not q_ids or not a_ids:
                raise ValueError(f"Empty tokenization at idx={idx}")

            input_ids = q_ids + a_ids
            # use -100 to ignore loss for prompt tokens
            labels = [-100] * len(q_ids) + a_ids

            return {
                'input_ids': input_ids,
                'labels': labels,
                'pixel_values': pixel_values,
                'sample_id': idx  # store sample ID for reference
            }
        except Exception as e:
            # only print errors occasionally to avoid flooding the console
            if random.random() < 0.05:  # Only print ~5% of errors
                print(f"Error processing sample {idx}: {e}")
            
            # return a dummy sample that can be detected and filtered
            return {
                'input_ids': [0],
                'labels': [0],
                'pixel_values': torch.zeros((3, 224, 224)),
                'sample_id': -1  # invalid sample ID -1
            }

class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # check bad samples
        features = [f for f in features if f['input_ids'] is not None and len(f['input_ids']) > 1]
        if len(features) == 0:
            raise ValueError("All samples in batch are invalid")
        
        max_len = max(len(f['input_ids']) for f in features)
        input_ids, labels, pixel_values = [], [], []
        
        for f in features:
            # pad sequences
            input_pad_len = max_len - len(f['input_ids'])
            label_pad_len = max_len - len(f['labels'])
            
            input_ids.append(f['input_ids'] + [self.tokenizer.pad_token_id] * input_pad_len)
            labels.append(f['labels'] + [-100] * label_pad_len)  # -100 for padding in labels
            pixel_values.append(f['pixel_values'])
            
        # generate attention mask (1 for real tokens, 0 for padding)
        attention_mask = [[1 if t != self.tokenizer.pad_token_id else 0 for t in seq] for seq in input_ids]
            
        return {
            'input_ids': torch.tensor(input_ids),
            'labels': torch.tensor(labels),
            'pixel_values': torch.stack(pixel_values),  
            'attention_mask': torch.tensor(attention_mask)
        }