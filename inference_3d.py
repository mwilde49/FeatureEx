#!/usr/bin/env python
"""
3D Medical Image Inference Pipeline

Performs inference on 3D medical images using the trained 3D segmentation model.
"""

import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Configuration
BASE_DIR = Path('C:/FeatureEx')
MODEL_PATH = BASE_DIR / 'models_3d' / 'best_resnet3d_segmentation.pth'
PREPROCESSED_DIR = BASE_DIR / 'preprocessed_3d_data'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 5

print(f"3D Medical Image Inference Pipeline")
print(f"=====================================\n")

# Load configuration
config_path = PREPROCESSED_DIR / 'preprocessing_config.json'
if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    print(f"Configuration loaded:")
    print(f"  Target shape: {tuple(config['target_shape'])}")
    print(f"  Channels: {config['target_shape'][3]}")
    print(f"  Device: {DEVICE}\n")

# ===============================================
# 3D ResNet Model Definition
# ===============================================

class ResNet3DSegmentation(nn.Module):
    """3D ResNet-based segmentation model."""

    def __init__(self, in_channels=2, num_classes=5):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # Final layer
        self.final_conv = nn.Conv3d(64, num_classes, kernel_size=1)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [self._make_residual_block(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(self._make_residual_block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def _make_residual_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Decoder
        x = self.decoder(x)
        x = self.final_conv(x)

        return x

# ===============================================
# Load Model
# ===============================================

print(f"Loading model from: {MODEL_PATH}")

if not MODEL_PATH.exists():
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print(f"Please train the model first using Test_FE_PCA_3D.ipynb")
    exit(1)

model = ResNet3DSegmentation(in_channels=2, num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print(f"Model loaded successfully!\n")

# ===============================================
# Inference Function
# ===============================================

def predict_image(img_path, label_path=None):
    """
    Predict segmentation for a single 3D image.

    Args:
        img_path: Path to preprocessed image (NIfTI)
        label_path: Optional path to ground truth label

    Returns:
        Dictionary with predictions and metrics
    """
    result = {
        'image': str(img_path.name),
        'timestamp': datetime.now().isoformat(),
    }

    try:
        # Load image
        img_nib = nib.load(img_path)
        img_data = img_nib.get_fdata()

        # Normalize
        img_min = img_data.min()
        img_max = img_data.max()
        if img_max > img_min:
            img_normalized = (img_data - img_min) / (img_max - img_min)
        else:
            img_normalized = img_data

        # Convert to tensor (channels, depth, height, width) = (2, 32, 512, 1024)
        img_tensor = torch.from_numpy(np.transpose(img_normalized, (3, 2, 0, 1))).float().unsqueeze(0)
        img_tensor = img_tensor.to(DEVICE)

        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            predictions = outputs.argmax(dim=1).squeeze(0)  # Remove batch dimension

        # Convert to numpy
        pred_array = predictions.cpu().numpy()

        # Save predictions
        pred_nib = nib.Nifti1Image(pred_array, img_nib.affine)
        pred_path = img_path.parent.parent / 'predictions' / f"{img_path.stem}_pred.nii.gz"
        pred_path.parent.mkdir(exist_ok=True)
        nib.save(pred_nib, pred_path)

        result['prediction_saved'] = str(pred_path)
        result['prediction_shape'] = tuple(pred_array.shape)
        result['unique_classes'] = sorted(list(np.unique(pred_array)))
        result['status'] = 'success'

        # If ground truth available, compute metrics
        if label_path and label_path.exists():
            label_nib = nib.load(label_path)
            label_data = label_nib.get_fdata()
            label_single = label_data[:, :, :, 0]

            # Flatten for metrics
            pred_flat = pred_array.flatten()
            label_flat = label_single.astype(int).flatten()

            # Compute IoU per class
            iou_scores = {}
            for cls in range(NUM_CLASSES):
                intersection = np.logical_and(pred_flat == cls, label_flat == cls).sum()
                union = np.logical_or(pred_flat == cls, label_flat == cls).sum()
                iou = intersection / union if union > 0 else 0
                iou_scores[f'class_{cls}'] = float(iou)

            result['iou_scores'] = iou_scores
            result['mean_iou'] = float(np.mean(list(iou_scores.values())))

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)

    return result

# ===============================================
# Batch Inference
# ===============================================

print(f"Starting inference on test set...\n")

images_dir = PREPROCESSED_DIR / 'images'
labels_dir = PREPROCESSED_DIR / 'labels'

image_files = sorted([f for f in images_dir.glob('*.nii.gz')])[:10]  # First 10 for testing

results = []
for idx, img_path in enumerate(image_files, 1):
    label_path = labels_dir / f"{img_path.stem}.nii.gz"

    print(f"[{idx}/{len(image_files)}] Predicting {img_path.name}...")
    result = predict_image(img_path, label_path)
    results.append(result)

    if result['status'] == 'success':
        if 'mean_iou' in result:
            print(f"  Mean IoU: {result['mean_iou']:.4f}")
        print(f"  Classes predicted: {result['unique_classes']}")
    else:
        print(f"  ERROR: {result.get('error', 'Unknown error')}")

# ===============================================
# Save Results
# ===============================================

results_path = BASE_DIR / 'inference_results_3d.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n\nInference complete!")
print(f"Results saved to: {results_path}")
print(f"\nSummary:")
successful = sum(1 for r in results if r['status'] == 'success')
print(f"  Successful: {successful}/{len(results)}")
if successful > 0:
    with_metrics = sum(1 for r in results if 'mean_iou' in r)
    if with_metrics > 0:
        mean_ious = [r['mean_iou'] for r in results if 'mean_iou' in r]
        print(f"  Mean IoU (avg): {np.mean(mean_ious):.4f}")
