import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim import AdamW
from vit_model import ViTTiny  
import math
import logging
import argparse


class Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Train a Vision Transformer model')
        
        # Dataset parameters
        self.parser.add_argument('--data_root', type=str, default='nabirds_split', 
                                help='Root directory of the dataset')
        self.parser.add_argument('--num_classes', type=int, default=555, 
                                help='Number of classes in the dataset')
        
        # Training parameters
        self.parser.add_argument('--batch_size', type=int, default=32, 
                                help='Batch size for training')
        self.parser.add_argument('--epochs', type=int, default=100, 
                                help='Number of epochs to train')
        self.parser.add_argument('--lr', type=float, default=1e-4, 
                                help='Learning rate')
        self.parser.add_argument('--weight_decay', type=float, default=0.05, 
                                help='Weight decay for optimizer')
        self.parser.add_argument('--warmup_epochs', type=int, default=5, 
                                help='Number of warmup epochs')
        
        # Model parameters
        self.parser.add_argument('--img_size', type=int, default=224, 
                                help='Image size for training')
        self.parser.add_argument('--embed_dim', type=int, default=192, 
                                help='Embedding dimension')
        self.parser.add_argument('--depth', type=int, default=12, 
                                help='Number of transformer blocks')
        self.parser.add_argument('--num_heads', type=int, default=3, 
                                help='Number of attention heads')
        
        # Hardware parameters
        self.parser.add_argument('--num_workers', type=int, default=4, 
                                help='Number of workers for data loading')
    
    def parse_args(self):
        return self.parser.parse_args()


# Setup logging
def setup_logger():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/training_log.txt"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


# Function to log parameters
def log_epoch_params(logger, epoch, **params):
    """Log epoch training parameters and metrics"""
    log_str = f"Epoch {epoch}/{params.get('total_epochs', '?')}"
    
    for key, value in params.items():
        if key == 'total_epochs':
            continue
        if isinstance(value, torch.Tensor):
            value = value.item() if value.numel() == 1 else "[tensor]"
        log_str += f", {key}: {value}"
    
    logger.info(log_str)


if __name__ == "__main__":
    args_parser = Args()
    args = args_parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger()
    
    # Log initial configuration
    logger.info(f"Training configuration: {vars(args)}")
    logger.info(f"Using device: {device}")
    
    # Data transforms
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((args.img_size + 32, args.img_size + 32)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets and loaders
    train_set = datasets.ImageFolder(os.path.join(args.data_root, "train"), transform=train_tfms)
    val_set = datasets.ImageFolder(os.path.join(args.data_root, "test"), transform=val_tfms)
    
    logger.info(f"Dataset sizes - Train: {len(train_set)}, Val: {len(val_set)}")
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model: Tiny ViT
    model = ViTTiny(
        img_size=args.img_size,
        patch_size=16,
        in_channels=3,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=4.0,
        dropout=0.1
    )

    model.to(device)
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # LR scheduler with warmup
    warmup_steps = args.warmup_epochs * len(train_loader)
    total_steps = args.epochs * len(train_loader)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        # cosine annealing
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    logger.info("Starting training")

    
    for epoch in range(args.epochs):
        model.train()
        total_loss, total_correct = 0, 0
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()

            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clip
            
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            batch_correct = (logits.argmax(1) == labels).sum().item()
            total_correct += batch_correct
        
       
        train_acc = total_correct / len(train_loader.dataset)
        train_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        
        # Evaluation
        model.eval()
        val_correct = 0
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, labels).item()
                val_correct += (logits.argmax(1) == labels).sum().item()
        
        val_acc = val_correct / len(val_loader.dataset)
        val_loss = val_loss / len(val_loader)
        
        # Log epoch results
        log_epoch_params(
            logger, 
            epoch+1,
            total_epochs=args.epochs,
            train_loss=f"{train_loss:.4f}",
            train_acc=f"{train_acc:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.4f}",
            lr=f"{current_lr:.6f}"
        )

    
    torch.save(model.state_dict(), "tinyvit_nba.pth")
    logger.info("Training completed. Model saved as tinyvit_nba.pth")