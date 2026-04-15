import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split    
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# 1. Setup paths
MAPPING_CSV = 'gz2_filename_mapping.csv'
HART16_CSV = 'gz2_hart16.csv'
IMAGE_DIR = 'images_gz2/images'

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print(f"Using device: {device}")

# 2. Load and clean data
print("\nLoading files...")
mapping_df = pd.read_csv(MAPPING_CSV)
hart16_df = pd.read_csv(HART16_CSV)

# Fix column names
if 'dr7objid' in hart16_df.columns:
    hart16_df.rename(columns={'dr7objid': 'objid'}, inplace=True)

# Join the tables
df = pd.merge(mapping_df, hart16_df, on='objid', how='inner')

# Filter by high confidence
def get_label(row):
    p_smooth = row.get('t01_smooth_or_features_a01_smooth_debiased', 0)
    p_features = row.get('t01_smooth_or_features_a02_features_or_disk_debiased', 0)
    p_edgeon = row.get('t02_edgeon_a04_yes_debiased', 0)
    p_spiral = row.get('t04_spiral_a08_spiral_debiased', 0)
    p_odd = row.get('t06_odd_a14_yes_debiased', 0)
    
    if p_smooth > 0.8: return 1  # Elliptical
    elif (p_features > 0.8) and (p_edgeon < 0.5) and (p_spiral > 0.8): return 0 # Spiral
    elif (p_features > 0.8) and (p_odd > 0.8): return 2 # Irregular
    else: return -1 # Skip

df['label'] = df.apply(get_label, axis=1)
df = df[df['label'] != -1]

# Separate classes to balance them
df_spiral = df[df['label'] == 0]
df_elliptical = df[df['label'] == 1]
df_irregular = df[df['label'] == 2]

# Match counts to the smallest class
n_irregular = len(df_irregular)
print(f"Balancing data to {n_irregular} images per class")

df_spiral_bal = df_spiral.sample(n=n_irregular, random_state=42)
df_elliptical_bal = df_elliptical.sample(n=n_irregular, random_state=42)

# Combine and mix
df_balanced = pd.concat([df_spiral_bal, df_elliptical_bal, df_irregular])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Find images on disk
valid_paths, valid_labels = [], []
for index, row in tqdm(df_balanced.iterrows(), total=df_balanced.shape[0], desc="Checking images"):
    full_path = os.path.join(IMAGE_DIR, f"{row['asset_id']}.jpg")
    if os.path.exists(full_path):
        valid_paths.append(full_path)
        valid_labels.append(row['label'])

# 3. Split 60/20/20
# Set aside 20% for test
train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    valid_paths, valid_labels, test_size=0.2, random_state=42, stratify=valid_labels
)

# Split 80% pool into train and val
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_val_paths, train_val_labels, test_size=0.25, random_state=42, stratify=train_val_labels
)

# 4. Augment images
# Training gets random rotations
train_transform = transforms.Compose([
    transforms.CenterCrop(200),
    transforms.Resize((224, 224)), 
    transforms.RandomRotation(180),                  
    transforms.RandomHorizontalFlip(),                 
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
])

# Validation/Test stays original
val_test_transform = transforms.Compose([
    transforms.CenterCrop(200),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/Test stays original
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Image loader class
class GalaxyDataset(Dataset):
    def __init__(self, p, l, t): self.p, self.l, self.t = p, l, t
    def __len__(self): return len(self.p)
    def __getitem__(self, i): 
        img = Image.open(self.p[i]).convert("RGB")
        return self.t(img), torch.tensor(self.l[i], dtype=torch.long)

# Setup loaders
train_loader = DataLoader(GalaxyDataset(train_paths, train_labels, train_transform), batch_size=32, shuffle=True)
val_loader = DataLoader(GalaxyDataset(val_paths, val_labels, val_test_transform), batch_size=32)
test_loader = DataLoader(GalaxyDataset(test_paths, test_labels, val_test_transform), batch_size=32)

# 5. Build the model with a dropout layer to prevent overconfidence
print("Loading ResNet50 with Dropout...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# replace the final layer but add a Dropout layer before it
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5), 
    nn.Linear(num_ftrs, 3)
)
model = model.to(device)

# Training settings
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 6. Training loop
NUM_EPOCHS = 30
print(f"\nTraining on {device}...")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss, train_correct, train_total = 0.0, 0, 0
    
    # Process batches
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
    # Check validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    # Print progress
    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total
    avg_loss = running_loss / len(train_loader)
    
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

# 7. Final results
print("\nTesting...")
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, lbls in tqdm(test_loader, desc="Testing"):
        imgs = imgs.to(device)
        _, predicted = torch.max(model(imgs), 1)
        y_true.extend(lbls.numpy())
        y_pred.extend(predicted.cpu().numpy())

# Stats report
class_names = ['Spiral', 'Elliptical', 'Irregular']
print("\nPerformance:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Save graph
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig('galaxy_results.png')
print("Saved galaxy_results.png")

torch.save(model.state_dict(), 'galaxy_model.pth')