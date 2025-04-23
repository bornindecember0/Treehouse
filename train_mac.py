import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from timm import create_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from vit_model import ViTTiny  
import math
from timm.data.mixup import Mixup
import logging
import json


# Hyperparameters
BATCH_SIZE = 32
NUM_CLASSES = 555 # for nba
EPOCHS = 100 
LR = 1e-4 
IMG_SIZE = 224
DATA_ROOT = "nabirds_split"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Transforms
# train_tfms = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
# ])
# val_tfms = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
# ])


# Enhanced Transforms
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),  # random crop with zoom
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # color augmentation
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),  # larger
    transforms.CenterCrop(IMG_SIZE),  # center crop
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Datasets and loaders
train_set = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_tfms)
val_set = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"), transform=val_tfms)
# val_set = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=val_tfms)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Model 1: pretrained Tiny ViT
#model = create_model("vit_tiny_patch16_224", pretrained=True, num_classes=NUM_CLASSES)


# Model 2: self-built Tiny ViT
model = ViTTiny(
    img_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=NUM_CLASSES,
    embed_dim=192, # 192
    depth=12,
    num_heads=3, # 3
    mlp_ratio=4.0,
    dropout=0.1
)

model.to(device)

#!!! below starts to diff from train.py, so it works on macOS
if __name__ == "__main__":
    model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.05) # ->0.05
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    warmup_epochs = 5
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = warmup_epochs * len(train_loader)


    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        # cosine annealing
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_correct = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clip
            
            optimizer.step()
            scheduler.step() # step per batch


            total_loss += loss.item()
            total_correct += (logits.argmax(1) == labels).sum().item()
           
        
        train_acc = total_correct / len(train_loader.dataset) 
        
        # print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f}")
        # scheduler.step()

        # Evaluation
        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                correct += (logits.argmax(1) == labels).sum().item()
        val_acc = correct / len(val_loader.dataset)

        print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")
        print(f"[Epoch {epoch+1}/{EPOCHS}] Val Acc: {val_acc:.4f}")
          

    torch.save(model.state_dict(), "tinyvit_nba.pth")
    print("Model saved")
