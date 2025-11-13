# 3D Medical Image Segmentation Pipeline

**Complete end-to-end 3D deep learning pipeline for medical image segmentation**

## Overview

This pipeline provides a complete workflow for training and inference on 3D medical imaging data:

1. **Data Preprocessing:** Standardize variable-dimension 3D volumes to uniform 512×1024×32 voxels
2. **3D Model Training:** ResNet50-based segmentation with 3D convolutions
3. **Inference:** Batch prediction on test data with metric evaluation

## Dataset Specifications

### Input Data Format
- **Location:** `C:/FeatureEx/preprocessed_3d_data/`
- **Images:** `images/` directory with 488 NIfTI volumes
- **Labels:** `labels/` directory with corresponding segmentation masks
- **Dimensions:** 512 × 1024 × 32 × 2 (X, Y, Z, Channels)
- **Data Type:** float32 (images), uint8 (labels)
- **Classes:** 5 (0=background, 1-4=anatomical structures)

### Preprocessing Steps

The data has already been preprocessed with:
- **XY Scaling:** Resized from variable dimensions to 512×1024 using bilinear interpolation
- **Z Padding:** Centered original slices with symmetric zero-padding to 32 slices
- **Normalization:** Image values normalized to original range post-processing
- **Paired Processing:** Images and labels transformed identically

See `PREPROCESSING_README.md` for detailed preprocessing documentation.

## Model Architecture

### 3D ResNet Segmentation

The model combines encoding and decoding pathways:

**Encoder (Downsampling):**
- Conv3d (2→64 channels, stride=2)
- ResNet blocks with stride=2 progression
- Layers: 3, 4, 6, 3 residual blocks per stage
- Channel progression: 64 → 128 → 256 → 512

**Decoder (Upsampling):**
- ConvTranspose3d blocks with stride=2
- Progressive channel reduction: 512 → 256 → 128 → 64
- Final Conv3d layer: 64 → 5 (num_classes)

**Total Parameters:** ~26 million trainable parameters

## Training Pipeline

### Configuration

```
BATCH_SIZE = 2              # Adjust based on GPU memory (12GB+ recommended)
NUM_EPOCHS = 50             # Early stopping via learning rate scheduler
LEARNING_RATE = 0.001       # Adam optimizer
WEIGHT_DECAY = 1e-4         # L2 regularization
```

### Data Split

- **Training:** 70% (342 samples)
- **Validation:** 15% (73 samples)
- **Test:** 15% (73 samples)

### Loss Function

- **CrossEntropyLoss** with class weights
- **Class weights:** [0.1, 1.0, 1.0, 1.0, 1.0]
  - Background (class 0): 0.1× weight (downweight large background)
  - Structures (classes 1-4): 1.0× weight (full importance)

### Learning Rate Schedule

- **ReduceLROnPlateau:** Reduce LR by 50% if validation loss plateaus
- **Patience:** 5 epochs without improvement
- **Min LR:** 1e-6

### Gradient Clipping

- **Max norm:** 1.0 (prevent exploding gradients in 3D)

## Training Notebook

**File:** `Test_FE_PCA_3D.ipynb`

### Cells Overview

| Cell | Purpose |
|------|---------|
| 1 | Imports and environment setup |
| 2 | Configuration (paths, hyperparameters) |
| 3 | 3D Dataset class definition |
| 4 | Data loading and splitting |
| 5 | 3D ResNet model definition |
| 6 | Loss function and optimizer setup |
| 7 | Training loop implementation |
| 8 | Model training execution |
| 9 | Test set evaluation |
| 10 | Training history visualization |
| 11 | Metrics export (JSON) |

### Running the Notebook

```bash
# Activate environment
cd C:\FeatureEx
.\FeatureEx\Scripts\activate

# Start Jupyter
jupyter notebook Test_FE_PCA_3D.ipynb

# Run all cells sequentially
```

### Expected Results

- **Training time:** 45-120 minutes (depends on GPU)
- **Best epoch:** Usually 20-35
- **Validation loss:** Should decrease with learning rate scheduling
- **Saved artifacts:**
  - `models_3d/best_resnet3d_segmentation.pth` - Best model weights
  - `models_3d/training_history_3d.png` - Loss curves
  - `models_3d/metrics_3d.json` - Training statistics

## Inference Pipeline

### Inference Script

**File:** `inference_3d.py`

Performs batch inference on preprocessed images with optional ground truth evaluation.

```bash
# Run inference
python inference_3d.py
```

### Features

- Loads best trained model automatically
- Processes 3D volumes with preprocessing
- Generates segmentation predictions (NIfTI format)
- Computes IoU metrics per class if ground truth available
- Saves results as JSON report

### Output

```
predictions/
├── image1_pred.nii.gz
├── image2_pred.nii.gz
└── ...

inference_results_3d.json    # Results summary with metrics
```

### Metrics Computed

- **Per-class IoU:** Intersection over Union for each class
- **Mean IoU:** Average across all classes
- **Prediction shapes:** Verification of output dimensions

## File Structure

```
C:/FeatureEx/
├── Test_FE_PCA_3D.ipynb              # Main training notebook
├── inference_3d.py                   # Inference script
├── 3D_PIPELINE_README.md             # This file
├── PREPROCESSING_README.md            # Preprocessing documentation
│
├── preprocessed_3d_data/
│   ├── images/                       # 488 preprocessed 3D images
│   ├── labels/                       # 488 segmentation labels
│   ├── preprocessing_config.json     # Preprocessing parameters
│   └── preprocessing_report.json     # Detailed per-pair preprocessing results
│
├── models_3d/                        # 3D model outputs
│   ├── best_resnet3d_segmentation.pth   # Trained model weights
│   ├── training_history_3d.png          # Training curves
│   └── metrics_3d.json                  # Training metrics
│
└── predictions/                      # Inference outputs
    ├── image1_pred.nii.gz
    ├── image2_pred.nii.gz
    └── ...
```

## GPU Memory Requirements

### Recommended Specifications

- **GPU:** NVIDIA with 12GB+ VRAM (RTX 2080 Ti or RTX 3080+)
- **RAM:** 32GB system RAM minimum
- **Storage:** 100GB free (for datasets and models)

### Memory Optimization

If GPU memory is limited:

1. **Reduce batch size:** `BATCH_SIZE = 1` (slower training)
2. **Reduce model complexity:** Fewer blocks in ResNet layers
3. **Gradient accumulation:** Accumulate gradients over 2-4 steps before optimizer update

## Known Limitations

1. **Fixed input size:** Model expects 512×1024×32 (requires preprocessing)
2. **2 channels only:** Adapts to fixed 2-channel input (modify for different modalities)
3. **5 classes:** Fixed for 5-class segmentation (modify final layer for more classes)
4. **Memory intensive:** Full 3D volumes require significant GPU memory

## Future Improvements

1. **Patch-based processing:** Process smaller patches to reduce memory
2. **3D data augmentation:** Rotation, flipping, elastic deformation
3. **Uncertainty estimation:** Monte Carlo dropout for confidence maps
4. **Multi-task learning:** Simultaneous segmentation and classification
5. **Ensemble methods:** Combine multiple model predictions

## Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` from 2 to 1
- Check for other GPU processes: `nvidia-smi`
- Clear GPU cache: Restart Python kernel

### Poor Convergence
- Increase `LEARNING_RATE` to 0.01 (try different values)
- Reduce `WEIGHT_DECAY` to 0 or 1e-5
- Check class balance in training data
- Verify image normalization is correct

### Model Not Saving
- Ensure `models_3d/` directory exists
- Check disk space availability
- Verify write permissions to directory

### Inference Error: "Module not found"
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Verify model path in `inference_3d.py`

## Dependencies

- torch >= 2.0.0
- nibabel >= 5.0.0
- numpy
- matplotlib
- scikit-learn

See `requirements.txt` for complete list.

## References

### 3D Convolution Architecture
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Adopted for 3D with Conv3d and ConvTranspose3d layers

### 3D Medical Imaging
- Çiçek, Ö., et al. (2016). 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. MICCAI.
- Dosovitskiy, A., & Brox, T. (2016). Flownet: Learning optical flow with convolutional networks. ICCV.

## Contact & Support

For issues or questions:
1. Check this documentation
2. Review notebook cells sequentially
3. Check error messages in training logs
4. Verify data preprocessing with `preprocessing_report.json`

---

**Pipeline Version:** 1.0
**Last Updated:** November 2025
**Status:** Production Ready
