# FeatureEx - Feature Extraction and PCA Analysis Project

## Project Overview

This project implements a machine learning pipeline for medical image analysis using a combination of deep learning (Convolutional Neural Networks) and traditional radiomics features. The workflow includes image classification, feature extraction using PCA (Principal Component Analysis), and integration with radiomic features for downstream gene expression inference.

## Dataset

- **Total Images**: 150 grayscale PNG images
- **Image Labels**: 3 classes
  - Class 1: Images 001-050 (50 images)
  - Class 2: Images 051-100 (50 images)
  - Class 3: Images 101-150 (50 images)
- **Image Format**: Grayscale PNG
- **Storage**: `C:/FeatureEx/images/`
- **Metadata**: `C:/FeatureEx/images/image_labels.csv`

## Project Structure

```
C:/FeatureEx/
├── Test_FE_PCA.ipynb          # Main Jupyter notebook
├── images/                     # Image dataset directory
│   ├── 001.png - 150.png      # Grayscale medical images
│   └── image_labels.csv       # Image paths and labels
├── requirements.txt            # Python dependencies
├── FeatureEx/                  # Virtual environment
└── README.md                   # This documentation
```

## Workflow Components

### 1. Environment Setup and Imports

**Key Libraries:**
- **PyTorch**: Deep learning framework for CNN implementation
- **torchvision**: Image transformations and preprocessing
- **pandas**: Data manipulation and CSV handling
- **PIL (Pillow)**: Image loading and processing
- **matplotlib**: Visualization of results
- **scikit-learn**: PCA, StandardScaler, preprocessing
- **numpy**: Numerical operations

### 2. Data Loading and Preprocessing

**CustomImageDataset Class:**
- Loads images from paths specified in CSV file
- Applies transformations (resize to 28x28, convert to tensor)
- Handles grayscale conversion
- Returns image-label pairs for training

**Data Transformations:**
```python
- Resize to 28x28 pixels
- Convert to PyTorch tensor
- Grayscale conversion
```

**Data Validation:**
- Missing image check to ensure all paths in CSV are valid
- Verifies file existence before training

### 3. Convolutional Neural Network (CNN) Architecture

**Purpose:** Extract deep learning features from medical images for classification.

**Network Architecture:**
- **Input**: 28x28 grayscale images
- **Convolutional Layers**: Extract spatial features
- **Pooling Layers**: Downsample feature maps
- **Fully Connected Layers**: Classification
- **Output**: 3-class predictions

**Training Process:**
- Train/Validation split for model evaluation
- Loss function: Cross-Entropy Loss (for multi-class classification)
- Optimizer: Adam or SGD
- Metrics: Training loss, validation loss, accuracy

**Outputs:**
- Trained model weights
- Training/validation loss curves
- Predictions vs. actual labels comparison

### 4. Feature Extraction with PCA

**Purpose:** Dimensionality reduction and feature extraction from CNN activations.

**Process:**
1. Extract features from intermediate CNN layers
2. Flatten high-dimensional feature maps
3. Apply StandardScaler for normalization
4. Use PCA to reduce dimensions while preserving variance
5. Generate low-dimensional feature representations

**Benefits:**
- Reduces computational complexity
- Removes redundant information
- Preserves most important variance in data
- Creates compact feature vectors for downstream analysis

### 5. Radiomics Feature Integration

**Purpose:** Combine traditional hand-crafted radiomics features with deep learning features.

**Radiomics Features:**
- Texture features (GLCM, GLRLM, etc.)
- Shape features
- First-order statistics
- Wavelet features

**Feature Fusion:**
- Concatenate CNN-derived features with radiomics features
- Create unified feature space for comprehensive image representation
- Enable multi-modal analysis

### 6. Gene Expression Inference

**Purpose:** Predict gene expression patterns from fused image features.

**Pipeline:**
1. **Load Data**: CNN features + radiomic features
2. **Feature Fusion**: Combine both feature types
3. **Model Training**: Train regression model to predict gene expression
4. **Correlation Analysis**: Identify top correlated genes
5. **Visualization**: Plot predictions vs. actual gene expression

**Key Functions:**
- `train_gene_inference_model()`: Trains model on fused features
- `get_top_correlated_genes()`: Identifies most predictive genes
- Visualization of gene expression predictions

### 7. Visualization and Analysis

**Generated Plots:**
- **Loss Curves**: Training vs. validation loss over epochs
- **Accuracy Curves**: Model performance tracking
- **Prediction Comparison**: Side-by-side predicted vs. actual labels
- **Gene Expression Plots**: Scatter plots of predicted vs. actual expression
- **PCA Components**: Visualization of principal components

## Installation and Setup

### Prerequisites
- Python 3.10 or higher
- Windows OS (paths configured for Windows)

### Installation Steps

1. **Clone/Download the project** to `C:/FeatureEx/`

2. **Create virtual environment:**
```bash
cd C:/FeatureEx
python -m venv FeatureEx
```

3. **Activate environment:**
```bash
FeatureEx\Scripts\activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

5. **Register Jupyter kernel:**
```bash
python -m ipykernel install --user --name=FeatureEx --display-name="Python (FeatureEx)"
```

## Usage

### Running the Notebook

1. **Open VS Code** and open the folder `C:/FeatureEx`

2. **Open the notebook** `Test_FE_PCA.ipynb`

3. **Select kernel**: Choose "Python (FeatureEx)" from kernel selector
   - Alternative: Use interpreter at `C:\FeatureEx\FeatureEx\Scripts\python.exe`

4. **Run cells sequentially** using Shift+Enter or the Run button

### Workflow Execution Order

1. **Import Libraries** (Cell 1)
2. **Load and Check Data** (Cells 7-8)
3. **Define Dataset Class** (Cell 15)
4. **Prepare Data Loaders** (Cell 19)
5. **Validate Image Paths** (Cell 22)
6. **Train CNN Model** (Multiple cells)
7. **Extract Features** (PCA section)
8. **Load Radiomics Features** (If available)
9. **Feature Fusion** (Combine CNN + radiomics)
10. **Gene Expression Inference** (Cell 112)
11. **Visualize Results** (Various visualization cells)

## Key Outputs

### Model Outputs
- Trained CNN model weights
- Extracted deep learning features
- PCA-transformed feature vectors
- Radiomics feature arrays

### Visualizations
- Training/validation loss and accuracy curves
- Confusion matrices (if implemented)
- PCA component plots
- Gene expression correlation plots
- Prediction scatter plots

### Data Outputs
- `cnn_features.csv`: Extracted CNN features
- `radiomic_features.csv`: Radiomics features
- Model predictions and evaluations

## Technical Details

### CNN Training Parameters
- **Image Size**: 28x28 pixels
- **Batch Size**: 32 (configurable)
- **Number of Classes**: 3
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam (typically)

### PCA Configuration
- **Purpose**: Dimensionality reduction
- **Input**: High-dimensional CNN features
- **Output**: Reduced feature space
- **Variance Preserved**: Typically 95%+

### Data Split
- Training set: ~70-80%
- Validation set: ~20-30%
- Random split with reproducible seed

## Troubleshooting

### Common Issues

**Issue: Kernel not found**
- Solution: Ensure virtual environment is activated and kernel is registered
- Run: `python -m ipykernel install --user --name=FeatureEx`

**Issue: Missing images error**
- Solution: Check that all PNG files exist in `C:/FeatureEx/images/`
- Run the missing image check cell (Cell 22)

**Issue: Path errors**
- Solution: Verify all paths use Windows format (`C:/FeatureEx/...`)
- Check that `image_labels.csv` has correct paths

**Issue: Out of memory**
- Solution: Reduce batch size in DataLoader
- Clear GPU cache: `torch.cuda.empty_cache()`

**Issue: Import errors**
- Solution: Reinstall requirements: `pip install -r requirements.txt`

## Future Enhancements

### Potential Improvements
- [ ] Implement data augmentation for better generalization
- [ ] Add cross-validation for robust evaluation
- [ ] Experiment with different CNN architectures
- [ ] Implement feature selection algorithms
- [ ] Add more comprehensive evaluation metrics
- [ ] Export model for deployment
- [ ] Create automated pipeline script

## Notes

### Design Considerations
- **Input Size**: Currently set to 28x28, may need adjustment for MRI slices
- **Multi-slice MRI**: Need to address how to handle 3D medical imaging data
- **Feature Fusion Strategy**: Multiple approaches possible (early/late fusion)

### Best Practices
- Always run cells sequentially from top to bottom
- Validate data paths before training
- Monitor GPU memory usage during training
- Save intermediate results for long training sessions
- Document any parameter changes

## Dependencies

See `requirements.txt` for complete list. Key dependencies:
- torch==2.9.0
- torchvision==0.24.0
- pandas==2.3.3
- matplotlib==3.10.7
- scikit-learn==1.7.2
- numpy==2.2.6
- pillow==12.0.0

## Contact and Support

For issues or questions:
- Check the troubleshooting section
- Review cell outputs for error messages
- Ensure all dependencies are correctly installed

## License

[Specify your license here]

## Acknowledgments

This project implements medical image analysis techniques combining:
- Deep learning (CNN) for automatic feature learning
- Traditional radiomics for hand-crafted features
- PCA for dimensionality reduction
- Gene expression inference for biological insights
