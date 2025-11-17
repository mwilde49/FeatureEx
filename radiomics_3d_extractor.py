"""
3D Radiomic Feature Extraction Module

Provides functions for extracting radiomic features from 3D medical images.
Handles 4D NIfTI files by converting to 3D for PyRadiomics compatibility.

Usage:
    from radiomics_3d_extractor import extract_features_from_image, load_radiomic_features

    # Extract features from a single image
    features = extract_features_from_image(
        image_path='path/to/image.nii.gz',
        label_path='path/to/label.nii.gz',
        structure_classes=[1, 2, 3, 4]
    )

    # Load pre-extracted features
    data = load_radiomic_features('path/to/radiomics_3d_features.pkl')
"""

import numpy as np
from pathlib import Path
import pickle
import json
from typing import Dict, List, Optional, Tuple
import SimpleITK as sitk
from radiomics import featureextractor
from tempfile import NamedTemporaryFile


def create_aoi_mask(label_data: np.ndarray, structure_classes: List[int]) -> np.ndarray:
    """
    Create binary attention mask for all non-background structures.

    Args:
        label_data: 3D label array
        structure_classes: List of structure class labels (e.g., [1, 2, 3, 4])

    Returns:
        Binary mask array (uint8) where all non-background labels are 1
    """
    mask = np.isin(label_data, structure_classes).astype(np.uint8)
    return mask


def validate_mask(mask_data: np.ndarray, min_voxels: int = 100) -> bool:
    """
    Validate mask has sufficient voxels.

    Args:
        mask_data: Binary mask array
        min_voxels: Minimum required voxels

    Returns:
        Boolean indicating validity
    """
    voxel_count = np.count_nonzero(mask_data)
    return voxel_count >= min_voxels


def extract_features_from_image(
    image_path: str,
    label_path: str,
    structure_classes: List[int] = None,
    extractor: featureextractor.RadiomicsFeatureExtractor = None,
) -> Optional[Dict]:
    """
    Extract radiomic features from a single 3D medical image.

    Handles 4D NIfTI files by converting to 3D. Creates a combined ROI mask
    from all non-background structures.

    Args:
        image_path: Path to 3D/4D NIfTI image file
        label_path: Path to 3D/4D NIfTI label file
        structure_classes: List of structure class labels to include in ROI.
                          Defaults to [1, 2, 3, 4]
        extractor: RadiomicsFeatureExtractor instance. If None, creates new extractor.

    Returns:
        Dictionary of extracted features, or None if extraction failed

    Example:
        >>> features = extract_features_from_image(
        ...     'image.nii.gz',
        ...     'label.nii.gz',
        ...     structure_classes=[1, 2, 3, 4]
        ... )
        >>> print(f"Extracted {len(features)} features")
    """
    if structure_classes is None:
        structure_classes = [1, 2, 3, 4]

    if extractor is None:
        extractor = featureextractor.RadiomicsFeatureExtractor()

    try:
        # Load data using SimpleITK
        image_sitk = sitk.ReadImage(str(image_path))
        label_sitk = sitk.ReadImage(str(label_path))

        # Convert to numpy arrays (SimpleITK returns in (C, Z, Y, X) format)
        image_array = sitk.GetArrayFromImage(image_sitk)
        label_array = sitk.GetArrayFromImage(label_sitk)

        # Handle 4D data: extract first channel to get 3D volume
        if image_array.ndim == 4:
            image_3d = image_array[0, :, :, :]  # (Z, Y, X)
            label_3d = label_array[0, :, :, :]
        else:
            image_3d = image_array
            label_3d = label_array

        # Create binary mask for all non-background structures
        mask_array = np.isin(label_3d, structure_classes).astype(np.uint32)

        # Validate mask
        if not validate_mask(mask_array):
            return None

        # Create 3D SITK images from numpy arrays
        image_3d_sitk = sitk.GetImageFromArray(image_3d)
        mask_3d_sitk = sitk.GetImageFromArray(mask_array)

        # Set proper geometry (use only first 3 dimensions of spacing/origin)
        spacing_3d = image_sitk.GetSpacing()[:3]
        origin_3d = image_sitk.GetOrigin()[:3]

        image_3d_sitk.SetSpacing(spacing_3d)
        image_3d_sitk.SetOrigin(origin_3d)
        mask_3d_sitk.SetSpacing(spacing_3d)
        mask_3d_sitk.SetOrigin(origin_3d)

        # Save temporary files
        with NamedTemporaryFile(suffix='.nii.gz', delete=False) as f_img:
            temp_img_path = f_img.name
        with NamedTemporaryFile(suffix='.nii.gz', delete=False) as f_mask:
            temp_mask_path = f_mask.name

        sitk.WriteImage(image_3d_sitk, temp_img_path)
        sitk.WriteImage(mask_3d_sitk, temp_mask_path)

        # Extract features
        features = extractor.execute(temp_img_path, temp_mask_path)

        # Clean up temp files
        Path(temp_img_path).unlink(missing_ok=True)
        Path(temp_mask_path).unlink(missing_ok=True)

        return dict(features)

    except Exception as e:
        return None


def load_radiomic_features(
    pickle_path: str,
) -> Dict:
    """
    Load pre-extracted radiomic features from pickle file.

    Args:
        pickle_path: Path to pickle file containing features

    Returns:
        Dictionary containing:
            - 'features_df': pandas DataFrame with all features
            - 'feature_names': List of feature column names
            - 'sample_ids': List of sample identifiers
            - 'metadata': Metadata dictionary with extraction info

    Example:
        >>> data = load_radiomic_features('radiomics_3d_features.pkl')
        >>> print(f"Loaded {len(data['features_df'])} samples")
        >>> print(f"Features: {len(data['feature_names'])}")
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    return data


def save_radiomic_features(
    features_df,
    feature_names: List[str],
    sample_ids: List[str],
    extraction_log: List[Dict],
    output_dir: str,
    structure_classes: List[int] = None,
) -> Dict[str, str]:
    """
    Save extracted radiomic features to multiple formats.

    Saves features as pickle, CSV, and configuration/log files for easy access
    and reproducibility.

    Args:
        features_df: pandas DataFrame with features
        feature_names: List of feature column names
        sample_ids: List of sample identifiers
        extraction_log: List of extraction log entries
        output_dir: Directory to save output files
        structure_classes: List of structure classes used (for metadata)

    Returns:
        Dictionary mapping file descriptions to output paths

    Example:
        >>> output_files = save_radiomic_features(
        ...     features_df, feature_names, sample_ids, log,
        ...     output_dir='radiomics_3d'
        ... )
        >>> for desc, path in output_files.items():
        ...     print(f"{desc}: {path}")
    """
    if structure_classes is None:
        structure_classes = [1, 2, 3, 4]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}

    # Save pickle file
    pickle_path = output_dir / 'radiomics_3d_features.pkl'
    pickle_data = {
        'features_df': features_df,
        'feature_names': feature_names,
        'sample_ids': sample_ids,
        'metadata': {
            'total_samples': len(features_df),
            'total_features': len(feature_names),
            'roi_type': 'combined_all_structures',
            'structure_classes': structure_classes,
            'extraction_log': extraction_log
        }
    }
    with open(pickle_path, 'wb') as f:
        pickle.dump(pickle_data, f)
    output_files['pickle'] = str(pickle_path)

    # Save CSV files
    csv_path = output_dir / 'radiomics_3d_features.csv'
    features_df.to_csv(csv_path, index=False)
    output_files['csv_with_ids'] = str(csv_path)

    # Save features only (for analysis)
    features_only_path = output_dir / 'radiomics_3d_features_only.csv'
    features_df[feature_names].to_csv(features_only_path, index=False)
    output_files['csv_features_only'] = str(features_only_path)

    # Save extraction log
    log_path = output_dir / 'extraction_log.json'
    with open(log_path, 'w') as f:
        json.dump(extraction_log, f, indent=2)
    output_files['extraction_log'] = str(log_path)

    # Save configuration summary
    config_summary = {
        'extraction_info': {
            'total_samples': len(features_df),
            'total_features': len(feature_names),
            'roi_type': 'combined_all_structures',
            'structure_classes': structure_classes
        },
        'feature_names': feature_names,
        'sample_ids': sample_ids,
        'output_files': {
            'pickle': 'radiomics_3d_features.pkl',
            'csv_with_ids': 'radiomics_3d_features.csv',
            'csv_features_only': 'radiomics_3d_features_only.csv',
            'extraction_log': 'extraction_log.json',
            'config': 'radiomics_3d_config.json'
        }
    }
    config_path = output_dir / 'radiomics_3d_config.json'
    with open(config_path, 'w') as f:
        json.dump(config_summary, f, indent=2)
    output_files['config'] = str(config_path)

    return output_files


if __name__ == '__main__':
    print("Radiomic Feature Extraction Module")
    print("=" * 50)
    print("\nAvailable functions:")
    print("  - extract_features_from_image(): Extract from single image")
    print("  - load_radiomic_features(): Load pre-extracted features")
    print("  - save_radiomic_features(): Save features to multiple formats")
    print("\nSee docstrings for usage examples.")
