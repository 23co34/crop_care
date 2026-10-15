import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import timm
from pathlib import Path
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = Path("/home/nyx/Desktop/project_a1/a12/papaya_split")
train_dir = data_dir / "train"
val_dir = data_dir / "val"

# 1. Custom Dataset for Albumentations
class PapayaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            for img_path in cls_dir.glob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label

# 2. Heavy Field Augmentations (Albumentations)
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3), # Simulates out-of-focus phone cameras
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5), # Forces model to look at whole leaf
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

train_dataset = PapayaDataset(train_dir, transform=train_transform)
val_dataset = PapayaDataset(val_dir, transform=val_transform)
num_classes = len(train_dataset.classes)
print(f"Classes: {train_dataset.classes}")

# 3. Handle Class Imbalance via WeightedRandomSampler
class_counts = list(Counter(train_dataset.labels).values())
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
sample_weights = torch.tensor([class_weights[t] for t in train_dataset.labels])
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# 4. DataLoader (Batch size 16 to fit B3 on 6GB VRAM with gradient accumulation later)
batch_size = 16
accumulation_steps = 2 # Effective batch size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 5. Scale up the Model (EfficientNet-B3)
model = timm.create_model("convnextv2_nano", pretrained=True, num_classes=num_classes)
model = model.to(device)

# 6. Label Smoothing Loss (Prevents overconfidence on ambiguous leaves)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ==========================================
# STAGE 1: Train Classifier Head Only
# ==========================================
print("\n--- STAGE 1: Freezing Backbone (Epochs 1-5) ---")
for param in model.parameters():
    param.requires_grad = False
for param in model.get_classifier().parameters():
    param.requires_grad = True

optimizer_stage1 = optim.AdamW(model.get_classifier().parameters(), lr=1e-3)
stage1_epochs = 5

for epoch in range(stage1_epochs):
    model.train()
    correct, total = 0, 0
    optimizer_stage1.zero_grad()
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            optimizer_stage1.step()
            optimizer_stage1.zero_grad()
            
        correct += outputs.max(1)[1].eq(labels).sum().item()
        total += labels.size(0)
    print(f"Stage 1 - Epoch [{epoch+1}/{stage1_epochs}] | Train Acc: {correct/total*100:.2f}%")

# ==========================================
# STAGE 2: End-to-End Fine Tuning
# ==========================================
print("\n--- STAGE 2: Unfreezing Backbone (Epochs 6-30) ---")
for param in model.parameters():
    param.requires_grad = True

stage2_epochs = 25
optimizer_stage2 = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
effective_steps = len(train_loader) // accumulation_steps
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_stage2, T_max=stage2_epochs * effective_steps, eta_min=1e-6)

best_val_acc = 0.0

for epoch in range(stage2_epochs):
    model.train()
    correct, total = 0, 0
    optimizer_stage2.zero_grad()
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            optimizer_stage2.step()
            scheduler.step()
            optimizer_stage2.zero_grad()
            
        correct += outputs.max(1)[1].eq(labels).sum().item()
        total += labels.size(0)
        
    # Validation Phase
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_correct += outputs.max(1)[1].eq(labels).sum().item()
            val_total += labels.size(0)
            
    val_acc = val_correct/val_total*100
    print(f"Stage 2 - Epoch [{epoch+1}/{stage2_epochs}] | Train Acc: {correct/total*100:.2f}% | Val Acc: {val_acc:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "a12_max_acc_convnext.pth")
        print(f"--> Saved new best model ({best_val_acc:.2f}%)")

print("Training Complete. Best weights saved to a12_max_acc_convnext.pth")