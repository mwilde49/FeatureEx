"""
ResNet-based model for medical image classification and feature extraction
Replaces SimpleCNN with a more powerful ResNet architecture
"""

import torch
import torch.nn as nn
from torchvision import models

class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-based model for classification and feature extraction.
    Uses ResNet18 backbone adapted for grayscale images.
    """
    def __init__(self, num_classes=3, feature_dim=512, pretrained=True):
        """
        Args:
            num_classes: Number of output classes (default: 3)
            feature_dim: Dimension of extracted features (default: 512)
            pretrained: Use ImageNet pretrained weights (default: True)
        """
        super(ResNetFeatureExtractor, self).__init__()

        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)

        # Modify first conv layer for grayscale (1 channel) input
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # New: Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the original fully connected layer
        num_features = self.resnet.fc.in_features  # 512 for ResNet18
        self.resnet.fc = nn.Identity()  # Remove classification head

        # Add custom feature extraction and classification layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape [batch_size, 1, H, W]
        Returns:
            logits: Classification logits [batch_size, num_classes]
            features: Extracted features [batch_size, feature_dim]
        """
        # Extract features using ResNet backbone
        x = self.resnet(x)  # [batch_size, 512]

        # Extract custom features
        features = self.feature_extractor(x)  # [batch_size, feature_dim]

        # Classification
        logits = self.classifier(features)  # [batch_size, num_classes]

        return logits, features

    def extract_features(self, x):
        """
        Extract features without classification
        Args:
            x: Input tensor of shape [batch_size, 1, H, W]
        Returns:
            features: Extracted features [batch_size, feature_dim]
        """
        x = self.resnet(x)
        features = self.feature_extractor(x)
        return features


class ResNet50FeatureExtractor(nn.Module):
    """
    ResNet50-based model for more powerful feature extraction.
    Use this for larger datasets or when you need more capacity.
    """
    def __init__(self, num_classes=3, feature_dim=512, pretrained=True):
        super(ResNet50FeatureExtractor, self).__init__()

        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)

        # Modify first conv layer for grayscale input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove original FC layer
        num_features = self.resnet.fc.in_features  # 2048 for ResNet50
        self.resnet.fc = nn.Identity()

        # Custom layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features

    def extract_features(self, x):
        x = self.resnet(x)
        features = self.feature_extractor(x)
        return features


# Training function compatible with ResNet model
def train_resnet_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, device='cuda'):
    """
    Train ResNet model for classification

    Args:
        model: ResNet model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cuda' or 'cpu')

    Returns:
        model: Trained model
        history: Dictionary containing training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    history = {
        'train_losses': [],
        'train_accuracies': [],
        'val_losses': [],
        'val_accuracies': []
    }

    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(images)  # Get logits, ignore features
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
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

        # Update learning rate
        scheduler.step(val_loss)

        # Store history
        history['train_losses'].append(train_loss)
        history['train_accuracies'].append(train_acc)
        history['val_losses'].append(val_loss)
        history['val_accuracies'].append(val_acc)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    return model, history


# Feature extraction function
def extract_features_from_resnet(model, dataloader, device='cuda'):
    """
    Extract features from trained ResNet model

    Args:
        model: Trained ResNet model
        dataloader: Data loader
        device: Device to use

    Returns:
        features: Numpy array of features [num_samples, feature_dim]
        labels: Numpy array of labels [num_samples]
    """
    model.eval()
    model = model.to(device)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            _, features = model(images)  # Get features

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    import numpy as np
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    return features, labels


# Example usage code snippet
"""
# Example: How to use in notebook

# 1. Create ResNet model
model = ResNetFeatureExtractor(num_classes=3, feature_dim=512, pretrained=True)

# 2. Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, history = train_resnet_model(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    lr=0.001,
    device=device
)

# 3. Extract features for downstream analysis
features, labels = extract_features_from_resnet(model, dataloader, device)

# 4. Apply PCA to features
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

pca = PCA(n_components=50)
features_pca = pca.fit_transform(features_scaled)

print(f"Original features shape: {features.shape}")
print(f"PCA features shape: {features_pca.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
"""
