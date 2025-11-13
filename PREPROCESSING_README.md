# 3D Data Preprocessing Pipeline

## Overview

This preprocessing pipeline standardizes all 3D medical images and labels to uniform dimensions to facilitate model training.

**Target Output Dimensions:** 512 × 1024 × 32 × 2
- X dimension: 512 (scaled from original)
- Y dimension: 1024 (scaled from original)
- Z dimension: 32 (padded with blank slices, centered)
- Channels: 2 (preserved from original)

## Key Features

### 1. Paired Processing
- Images and labels are processed together as pairs
- Both use identical dimensions after preprocessing
- Alignment is guaranteed since they share the same transformation parameters

### 2. XY Dimension Scaling
- **Method:** PIL Image.resize() with interpolation
- **Images:** Bilinear interpolation (preserves intensity values)
- **Labels:** Nearest neighbor interpolation (preserves class labels)
- **Efficiency:** Per-slice 2D processing reduces memory consumption

### 3. Z Dimension Padding
- **Method:** Symmetric zero-padding
- **Approach:** Centers original slices with blank padding
- **Calculation:**
  - If original Z < 32: Add padding before and after to center slices
  - If original Z ≥ 32: Extract middle 32 slices
- **Formula:** `pad_before = (target_z - curr_z) // 2`, `pad_after = target_z - curr_z - pad_before`

### 4. Memory Efficiency
- Processes one 2D slice at a time
- Uses PIL for efficient image resizing
- Avoids large intermediate array allocation
- Handles datasets with variable input dimensions

## File Structure

```
preprocessed_3d_data/
├── images/              # Preprocessed image volumes
│   ├── 1.5L.nii.gz
│   ├── 1.6L.nii.gz
│   └── ...
├── labels/              # Preprocessed segmentation labels
│   ├── 1.5L.nii.gz
│   ├── 1.6L.nii.gz
│   └── ...
├── preprocessing_config.json    # Configuration and statistics
└── preprocessing_report.json     # Detailed results per pair
```

## Processing Scripts

### `preprocess_3d_data_v2.py`
Main preprocessing script with memory-efficient slice-based processing.

**Features:**
- Slice-by-slice processing
- Paired image-label transformation
- Verification checks after preprocessing
- Progress reporting (every 50 pairs)
- JSON configuration export

**Usage:**
```bash
python preprocess_3d_data_v2.py
```

**Output:**
- Preprocessed NIfTI files in `preprocessed_3d_data/images/` and `labels/`
- `preprocessing_config.json` with summary statistics
- `preprocessing_report.json` with per-pair details

## Data Format Details

### Original Format
- **Shape:** (X, Y, Z, channels)
- **X range:** 240-1024
- **Y range:** 120-512
- **Z range:** 12-50
- **Channels:** 2 (multi-modal)

### Preprocessed Format
- **Shape:** (512, 1024, 32, 2)
- **Data type (images):** float32
- **Data type (labels):** uint8
- **Values (images):** Denormalized to original range
- **Values (labels):** 0 (background), 1-4 (structure classes)

## Quality Checks

Each preprocessed pair undergoes verification:

1. **Shape Verification:** Image and label shapes must match exactly
2. **Dimension Verification:** Output shape must be (512, 1024, 32, 2)
3. **Label Value Validation:** Only values 0-4 allowed
4. **NaN Detection:** No NaN values permitted
5. **Infinity Detection:** No infinite values permitted

## Processing Results

Expected statistics:
- **Input pairs:** 488 (from 3D dataset analysis)
- **Expected successful:** ~480-488 (>98%)
- **Typical failure causes:**
  - Corrupted NIfTI files
  - Inconsistent shapes between paired images and labels
  - Invalid label values

## Integration with Training

### Using Preprocessed Data

1. **Update data loading:**
   ```python
   from pathlib import Path
   import nibabel as nib

   preprocessed_dir = Path('C:/FeatureEx/preprocessed_3d_data')
   images_dir = preprocessed_dir / 'images'
   labels_dir = preprocessed_dir / 'labels'

   # Load image
   img = nib.load(images_dir / f'{sample_name}.nii.gz')
   img_data = img.get_fdata()  # Shape: (512, 1024, 32, 2)

   # Load label
   lbl = nib.load(labels_dir / f'{sample_name}.nii.gz')
   lbl_data = lbl.get_fdata()  # Shape: (512, 1024, 32, 2)
   ```

2. **PyTorch Dataset:**
   ```python
   class Preprocessed3DDataset(Dataset):
       def __init__(self, images_dir, labels_dir):
           self.images = sorted(Path(images_dir).glob('*.nii.gz'))
           self.labels = sorted(Path(labels_dir).glob('*.nii.gz'))

       def __getitem__(self, idx):
           img = nib.load(self.images[idx]).get_fdata()
           lbl = nib.load(self.labels[idx]).get_fdata()

           # Convert to torch tensors
           img_tensor = torch.from_numpy(img).float()
           lbl_tensor = torch.from_numpy(lbl.squeeze()).long()

           return img_tensor, lbl_tensor
   ```

3. **Model Input:**
   - Expected input shape: `(batch_size, 512, 1024, 32, 2)`
   - Use 3D convolutions: `torch.nn.Conv3d(2, 64, kernel_size=3, ...)`
   - Output channels: 5 (for 5-class segmentation)

## Troubleshooting

### Issue: Processing hangs or crashes
**Solution:** Memory may be insufficient. Check available RAM; PILbased processing is memory-efficient but still requires workspace.

### Issue: Some pairs fail verification
**Solution:** Check `preprocessing_report.json` for specific issues. May need to manually inspect failed images.

### Issue: Output shapes inconsistent
**Solution:** Verify all preprocessed files with:
```bash
python verify_preprocessing.py
```

## Performance Metrics

Typical processing time:
- **Per image:** 2-4 seconds (depends on original dimensions)
- **Full dataset (488 pairs):** ~20-30 minutes
- **Memory usage:** ~500 MB peak (per-image + temporary buffers)

## Next Steps

After successful preprocessing:

1. **Update Test_FE_PCA_3D.ipynb** to use preprocessed data
2. **Adapt DataLoader** for new preprocessed data structure
3. **Retrain ResNet models** with 3D convolutions and proper input dimensions
4. **Validate output** on test set with standardized input size
5. **Compare performance** with original variable-size inputs

## Configuration Export

The `preprocessing_config.json` contains:
```json
{
  "status": "complete",
  "total_pairs": 488,
  "successful_pairs": 487,
  "failed_pairs": 1,
  "success_rate": 99.8,
  "target_shape": [512, 1024, 32, 2],
  "output_directories": {
    "images": "C:/FeatureEx/preprocessed_3d_data/images",
    "labels": "C:/FeatureEx/preprocessed_3d_data/labels"
  }
}
```

---

**Ready to preprocess?** Run `preprocess_3d_data_v2.py` to begin.

**Questions?** Refer to specific sections above or check the detailed report in `preprocessing_report.json`.
