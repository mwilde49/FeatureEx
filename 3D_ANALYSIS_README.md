# 3D Data Analysis for Pipeline Configuration

## Overview

This document describes how to run the 3D data analysis notebook to extract all necessary configuration parameters for adapting the FeatureEx pipeline to 3D multichannel, multiclass data.

## Dataset Location

Your 3D NIfTI files are located in:
- **Images:** `C:/FeatureEx/imagesTr/imagesTr/`
- **Labels:** `C:/FeatureEx/labelsTr/labelsTr/`

## Analysis Notebook

**File:** `analyze_3d_data.ipynb`

This notebook will analyze all your 3D data and extract:

### 1. Dataset Information
- ✅ Number of 3D images
- ✅ Number of segmentation labels
- ✅ Image-label matching verification
- ✅ Data consistency checks

### 2. Image Dimensions
- ✅ Number of channels (multi-channel support)
- ✅ Depth (Z dimension)
- ✅ Height (Y dimension)
- ✅ Width (X dimension)
- ✅ Shape consistency across dataset
- ✅ Handling of inconsistent dimensions

### 3. Label Information
- ✅ Unique label values found
- ✅ Number of classes (including background)
- ✅ Label distribution
- ✅ Multi-class segmentation support
- ✅ Background label detection

### 4. Data Characteristics
- ✅ Data types (float32, int32, etc.)
- ✅ Value ranges (min, max, mean, std)
- ✅ Data quality checks

### 5. Memory Requirements
- ✅ Per-image memory footprint
- ✅ Total dataset size
- ✅ Recommended batch sizes
- ✅ GPU memory requirements

### 6. Configuration Export
- ✅ JSON configuration file (`3d_dataset_config.json`)
- ✅ All parameters needed for pipeline setup

## How to Run

### Step 1: Activate Environment
```bash
cd C:\FeatureEx
.\FeatureEx\Scripts\activate
```

### Step 2: Install Dependencies (if needed)
```bash
pip install -r requirements.txt
```

### Step 3: Run Jupyter
```bash
jupyter notebook analyze_3d_data.ipynb
```

### Step 4: Execute All Cells
Run all cells sequentially to generate:
1. Complete dataset analysis
2. Configuration summary printed to console
3. JSON configuration file saved to disk

## Output

The notebook will generate:

### 1. Console Output
- Dataset statistics
- Dimension analysis
- Label distribution
- Memory analysis
- Configuration summary

### 2. JSON Configuration File
**File:** `3d_dataset_config.json`

This file contains all extracted parameters in a structured format:
```json
{
  "dataset_info": {
    "num_images": <count>,
    "num_labels": <count>,
    "matching_pairs": <count>
  },
  "image_dimensions": {
    "channels": <count>,
    "depth": <size>,
    "height": <size>,
    "width": <size>
  },
  "labels": {
    "unique_labels": [<list>],
    "num_classes": <count>,
    "has_background": <bool>
  },
  "data_types": {
    "image_dtype": "<dtype>",
    "label_dtype": "<dtype>"
  },
  "memory_mb": {
    "per_image": <float>,
    "total_dataset": <float>
  }
}
```

## What to Look For

### Configuration Parameters
After running the analysis, note these key parameters:

1. **Number of Classes (num_classes)**
   - Used for: Model output layer size
   - Example: 4 classes (1 background + 3 structures)

2. **Image Dimensions (channels, depth, height, width)**
   - Used for: Input layer configuration
   - Example: (2, 128, 256, 256) = 2 channels, 128×256×256

3. **Label Mapping**
   - Used for: Loss function and metrics
   - Maps: 0=background, 1=class1, 2=class2, etc.

4. **Data Types**
   - Used for: Tensor conversion and processing
   - Example: float32 for images, int32 for labels

5. **Memory Requirements**
   - Used for: Batch size selection
   - Plan: GPU memory / per-image MB = max batch size

## Next Steps

Once you have the analysis output, you will:

1. **Update Test_FE_PCA_3D.ipynb** with configuration parameters
2. **Adapt image loading** for 3D NIfTI files
3. **Modify ResNet architecture** to 3D convolutions
4. **Update data preprocessing** for 3D volumes
5. **Configure classification** for extracted classes
6. **Set up radiomics extraction** for 3D data
7. **Train new 3D models** with configuration

## Troubleshooting

### Error: ModuleNotFoundError: No module named 'nibabel'
Run: `pip install nibabel`

### Error: No files found in imagesTr
Check that files are in: `C:/FeatureEx/imagesTr/imagesTr/` (note the nested directory structure)

### Error: Image and label shape mismatch
The notebook will report this - may indicate preprocessing needed

### Memory Issues When Loading All Files
The notebook loads one file at a time to minimize memory usage

## Key Dependencies

- **nibabel** - NIfTI file I/O
- **numpy** - Array operations
- **pandas** - Data analysis
- **torch** - Deep learning (for eventual model adaptation)
- **torchvision** - Vision utilities

All are included in `requirements.txt`

---

**Ready to analyze?** Run the notebook and you'll have complete configuration for your 3D pipeline!
