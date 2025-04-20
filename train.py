import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from timm import create_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Hyperparameters
BATCH_SIZE = 32
NUM_CLASSES = 200
EPOCHS = 10
LR = 1e-4
IMG_SIZE = 224
DATA_ROOT = "cub_split"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Datasets and loaders
train_set = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_tfms)
val_set = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=val_tfms)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Model: Tiny ViT
model = create_model("vit_tiny_patch16_224", pretrained=True, num_classes=NUM_CLASSES)
model.to(device)

# Optimizer, loss, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

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
        optimizer.step()

        total_loss += loss.item()
        total_correct += (logits.argmax(1) == labels).sum().item()

    train_acc = total_correct / len(train_loader.dataset)
    print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f}")
    scheduler.step()

    # Evaluation
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            correct += (logits.argmax(1) == labels).sum().item()
    val_acc = correct / len(val_loader.dataset)
    print(f"[Epoch {epoch+1}] Validation Accuracy: {val_acc:.4f}")

torch.save(model.state_dict(), "tinyvit_cub200.pth")
print("Model saved")
