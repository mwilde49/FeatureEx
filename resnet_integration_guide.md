# ResNet Integration Guide

## How to Replace SimpleCNN with ResNet in Test_FE_PCA.ipynb

### Step 1: Import ResNet Model

Add this cell after your imports (around Cell 1):

```python
# Import ResNet model
from resnet_model import (
    ResNetFeatureExtractor,
    ResNet50FeatureExtractor,
    train_resnet_model,
    extract_features_from_resnet
)
```

Alternatively, add the ResNet class directly in a cell:

```python
import torch
import torch.nn as nn
from torchvision import models

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, num_classes=3, feature_dim=512, pretrained=True):
        super(ResNetFeatureExtractor, self).__init__()

        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)

        # Modify first conv layer for grayscale (1 channel) input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the original fully connected layer
        num_features = self.resnet.fc.in_features  # 512 for ResNet18
        self.resnet.fc = nn.Identity()

        # Custom feature extraction and classification layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # Extract features using ResNet backbone
        x = self.resnet(x)

        # Extract custom features
        features = self.feature_extractor(x)

        # Classification
        logits = self.classifier(features)

        return logits, features
```

### Step 2: Update Image Size (IMPORTANT)

ResNet expects larger images. Update your transform in Cell 19:

**OLD:**
```python
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])
```

**NEW:**
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet standard size
    transforms.ToTensor(),
    # Optional: normalize like ImageNet (recommended for pretrained models)
    transforms.Normalize(mean=[0.485], std=[0.229])  # Single channel
])
```

### Step 3: Replace Model Creation (Cell 24)

**OLD:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**NEW:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create ResNet model
model = ResNetFeatureExtractor(
    num_classes=3,
    feature_dim=512,  # Can adjust: 128, 256, 512, 1024
    pretrained=True   # Use ImageNet pretrained weights
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower LR for pretrained model

# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Step 4: Update Training Loop

Add scheduler step in your validation section:

```python
for epoch in range(num_epochs):
    # ... training code ...

    # After validation
    val_loss = total_val_loss / len(val_loader)
    val_acc = 100.0 * val_correct / val_total

    # Update learning rate based on validation loss
    scheduler.step(val_loss)

    # ... rest of code ...
```

### Step 5: Feature Extraction with ResNet

Replace your feature extraction code with:

```python
# Extract features using trained ResNet
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

all_features = []
all_labels = []

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        logits, features = model(images)  # Get both predictions and features

        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())

# Combine all batches
import numpy as np
features = np.vstack(all_features)
labels = np.concatenate(all_labels)

print(f"Extracted features shape: {features.shape}")
print(f"Features from {len(labels)} images")
```

### Step 6: Apply PCA to ResNet Features

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply PCA
n_components = 50  # or use 0.95 for variance-based selection
pca = PCA(n_components=n_components)
features_pca = pca.fit_transform(features_scaled)

print(f"Original features: {features.shape}")
print(f"PCA features: {features_pca.shape}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# Plot explained variance
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.bar(range(1, 11), pca.explained_variance_ratio_[:10])
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Top 10 Components')
plt.grid(True)
plt.tight_layout()
plt.show()
```

## Key Differences: SimpleCNN vs ResNet

| Feature | SimpleCNN | ResNet18 | ResNet50 |
|---------|-----------|----------|----------|
| Input Size | 28x28 | 224x224 | 224x224 |
| Parameters | ~50K | ~11M | ~23M |
| Feature Dim | 64 | 512 | 2048 |
| Depth | 5 layers | 18 layers | 50 layers |
| Pretrained | No | Yes (ImageNet) | Yes (ImageNet) |
| Training Time | Fast | Medium | Slower |
| Accuracy | Good | Better | Best |

## Recommendations

### For Small Dataset (150 images):
- **Use ResNet18** with pretrained weights
- **Feature dim**: 256-512
- **Freeze early layers**: Consider freezing early ResNet layers to prevent overfitting
- **Data augmentation**: Essential with small datasets

```python
# Freeze early layers
for name, param in model.resnet.named_parameters():
    if 'layer4' not in name and 'fc' not in name:
        param.requires_grad = False
```

### For Data Augmentation:
Add to your transforms:

```python
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])
```

## Complete Training Example

```python
# 1. Update transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# 2. Recreate dataset and dataloader
dataset = CustomImageDataset('C:/FeatureEx/images/image_labels.csv', transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 3. Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetFeatureExtractor(num_classes=3, feature_dim=512, pretrained=True).to(device)

# 4. Train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

num_epochs = 50
best_val_acc = 0

for epoch in range(num_epochs):
    # Training...
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss /= len(train_loader)
    train_acc = 100.0 * train_correct / train_total

    # Validation...
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            loss = criterion(logits, labels)

            val_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100.0 * val_correct / val_total

    scheduler.step(val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_resnet_model.pth')

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

print(f"Best validation accuracy: {best_val_acc:.2f}%")
```

## Troubleshooting

### Out of Memory Error:
- Reduce batch size: `batch_size=8` or `batch_size=4`
- Use smaller feature dim: `feature_dim=256`
- Use CPU: `device='cpu'`

### Poor Performance:
- Increase epochs: `num_epochs=100`
- Add data augmentation
- Try different learning rates: `lr=0.0001` to `lr=0.001`
- Freeze fewer layers or unfreeze all layers

### Model Not Learning:
- Check if using pretrained weights: `pretrained=True`
- Verify labels are 0-indexed (not 1-indexed)
- Check data normalization
- Verify image dimensions: should be `[batch, 1, 224, 224]`
