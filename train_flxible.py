# This is the train file for the multiple settings when training vit

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim import AdamW, SGD
from vit_model import ViTTiny  
import math
import logging
import argparse
from typing import Dict, Any, Optional
import json
from datetime import datetime


class Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='VIT training')
        self.parser.add_argument('--data_root', type=str, default='cub_split', 
                                help='Root directory of the dataset')   # cub -> 200 nab -> 555
        self.parser.add_argument('--num_classes', type=int, default=200, 
                                help='Number of classes in the dataset')
        self.parser.add_argument('--epochs', type=int, default=100, 
                                help='Number of epochs to train')
        self.parser.add_argument('--optimizer', type=str, default='adamw', 
                                choices=['adamw', 'sgd'],
                                help='Optimizer to use')
        self.parser.add_argument('--lr', type=float, default=1e-4, 
                                help='Learning rate')
        self.parser.add_argument('--min_lr', type=float, default=1e-6, 
                                help='Minimum learning rate for cosine schedule')
        self.parser.add_argument('--use_scheduler', type=bool, default=True, 
                                help='Use cosine scheduler or constant LR')
        self.parser.add_argument('--warmup_epochs', type=int, default=5, 
                                help='Number of warmup epochs')
        self.parser.add_argument('--batch_size', type=int, default=32, 
                                help='Batch size for training')
        self.parser.add_argument('--val_batch_size', type=int, default=64, 
                                help='Batch size for validation')
        self.parser.add_argument('--num_workers', type=int, default=16, 
                                help='Number of workers for data loading')
        self.parser.add_argument('--weight_decay', type=float, default=0.05, 
                                help='Weight decay (L2 regularization)')
        self.parser.add_argument('--dropout', type=float, default=0.1, 
                                help='Dropout rate')
        self.parser.add_argument('--grad_clip', type=float, default=1.0, 
                                help='Gradient clipping value (0 for no clipping)')
        self.parser.add_argument('--img_size', type=int, default=224, 
                                help='Image size for training')
        self.parser.add_argument('--patch_size', type=int, default=16, 
                                help='Patch size for ViT')
        self.parser.add_argument('--embed_dim', type=int, default=192, 
                                help='Embedding dimension')
        self.parser.add_argument('--depth', type=int, default=12, 
                                help='Number of transformer blocks')
        self.parser.add_argument('--num_heads', type=int, default=3, 
                                help='Number of attention heads')
        self.parser.add_argument('--augmentation', type=str, default='standard', 
                                choices=['standard', 'advanced', 'minimal'],
                                help='Data augmentation strategy')
        self.parser.add_argument('--mixup_alpha', type=float, default=0.0, 
                                help='Mixup alpha (0 to disable)')
        self.parser.add_argument('--cutmix_alpha', type=float, default=0.0, 
                                help='CutMix alpha (0 to disable)')
        
     
        self.parser.add_argument('--global_pool', type=str, default='cls', 
                                choices=['cls', 'avg', 'weighted', 'cls+avg', 'cls+weighted'],
                                help='Feature pooling strategy for ViT')
        self.parser.add_argument('--exp_name', type=str, default=None, 
                                help='Experiment name for logging')
        self.parser.add_argument('--output_dir', type=str, default='experiments', 
                                help='Directory to save experiment outputs')
    
    def parse_args(self):
        return self.parser.parse_args()


class ExperimentLogger:
    def __init__(self, args):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = args.exp_name or f"exp_{timestamp}"
        self.exp_dir = os.path.join(args.output_dir, self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        self.logger = logging.getLogger(self.exp_name)
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.exp_dir, 'training.log'))
        fh.setLevel(logging.INFO)
        
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
        with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
    
    def log(self, message):
        self.logger.info(message)
    
    def log_epoch(self, epoch, metrics):
        msg = f"Epoch {epoch}: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.logger.info(msg)


def get_data_augmentation(args):
    if args.augmentation == 'minimal':
        train_transforms = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    elif args.augmentation == 'standard':
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    else:  # more advanced
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((args.img_size + 32, args.img_size + 32)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms


def get_optimizer(model, args):
    if args.optimizer == 'adamw':
        return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:  # sgd
        return SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, 
                   momentum=0.9, nesterov=True)


def get_scheduler(optimizer, args, total_steps):
    if not args.use_scheduler:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    
    # cosine with warmup
    warmup_steps = args.warmup_epochs * total_steps // args.epochs if args.warmup_epochs > 0 else 0
    
    def cosine_schedule_with_warmup(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine_decay * (1 - args.min_lr / args.lr) + args.min_lr / args.lr
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_schedule_with_warmup)


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).sample()
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(model, train_loader, criterion, optimizer, device, args):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        if args.mixup_alpha > 0 and torch.rand(1).item() < 0.5:  # mixup 50% of the time
            imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, args.mixup_alpha)
            logits = model(imgs)
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item() * imgs.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += imgs.size(0)
    
    return total_loss / total_samples, total_correct / total_samples


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += imgs.size(0)
    
    return total_loss / total_samples, total_correct / total_samples


def main():
    args = Args().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_logger = ExperimentLogger(args)
    exp_logger.log(f"Using device: {device}")
    exp_logger.log(f"Configuration: {json.dumps(vars(args), indent=2)}")
    train_transforms, val_transforms = get_data_augmentation(args)
    train_set = datasets.ImageFolder(os.path.join(args.data_root, "train"), transform=train_transforms)
    val_set = datasets.ImageFolder(os.path.join(args.data_root, "val"), transform=val_transforms)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, 
                           num_workers=args.num_workers, pin_memory=True)
    
    exp_logger.log(f"Dataset sizes - Train: {len(train_set)}, Val: {len(val_set)}")
    
    # Model
    model = ViTTiny(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=3,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=4.0,
        dropout=args.dropout,
        global_pool=args.global_pool 
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args, len(train_loader) * args.epochs)
    
    # training loop
    best_val_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, args)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr
        }
        exp_logger.log_epoch(epoch + 1, metrics)
    
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'args': vars(args)
    }, os.path.join(exp_logger.exp_dir, 'final_model.pth'))
    
    exp_logger.log(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
