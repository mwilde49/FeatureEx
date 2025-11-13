#!/usr/bin/env python
"""
3D Medical Image Preprocessing Pipeline (v2 - Memory Efficient)

Standardizes 3D medical images and labels to uniform dimensions:
- X (height): Scaled to 512
- Y (width): Scaled to 1024
- Z (depth): Padded with blank slices to 32 (centered)

Key Improvements:
- Correctly handles actual data format: (X, Y, Z, channels)
- Memory-efficient processing using PIL for resizing
- Paired processing ensures images and labels remain aligned
- Symmetrical Z-axis padding centers original slices
"""

import nibabel as nib
import numpy as np
from PIL import Image
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ======================
# Configuration
# ======================
BASE_DIR = Path('C:/FeatureEx')
IMAGES_DIR = BASE_DIR / 'imagesTr' / 'imagesTr'
LABELS_DIR = BASE_DIR / 'labelsTr' / 'labelsTr'

OUTPUT_DIR = BASE_DIR / 'preprocessed_3d_data'
IMAGES_OUTPUT = OUTPUT_DIR / 'images'
LABELS_OUTPUT = OUTPUT_DIR / 'labels'

# Target dimensions (matching actual data structure: X, Y, Z, channels)
TARGET_X = 512
TARGET_Y = 1024
TARGET_Z = 32

print("="*70)
print("3D MEDICAL IMAGE PREPROCESSING PIPELINE (V2 - Memory Efficient)")
print("="*70)
print(f"\nConfiguration:")
print(f"  Target dimensions: X={TARGET_X}, Y={TARGET_Y}, Z={TARGET_Z}")
print(f"  Data format: (X, Y, Z, channels)")
print(f"  Input images: {IMAGES_DIR}")
print(f"  Input labels: {LABELS_DIR}")
print(f"  Output directory: {OUTPUT_DIR}")

# Create output directories
IMAGES_OUTPUT.mkdir(parents=True, exist_ok=True)
LABELS_OUTPUT.mkdir(parents=True, exist_ok=True)
print(f"\nOutput directories created:")
print(f"  {IMAGES_OUTPUT}")
print(f"  {LABELS_OUTPUT}")

# ======================
# Helper Functions
# ======================

def get_file_pairs():
    """Get all image-label file pairs."""
    image_files = sorted([f for f in IMAGES_DIR.glob('*.nii*')])
    label_files = sorted([f for f in LABELS_DIR.glob('*.nii*')])

    image_basenames = {f.stem: f for f in image_files}
    label_basenames = {f.stem: f for f in label_files}

    matching_pairs = set(image_basenames.keys()) & set(label_basenames.keys())

    pairs = [
        (image_basenames[name], label_basenames[name])
        for name in sorted(matching_pairs)
    ]

    return pairs

def pad_z_axis(data, target_z):
    """
    Pad Z-axis with blank slices, centering original data.

    Args:
        data: Array with shape (X, Y, Z, channels) or (X, Y, Z)
        target_z: Target Z dimension

    Returns:
        Padded array
    """
    if data.ndim == 4:
        x, y, curr_z, channels = data.shape
    else:
        x, y, curr_z = data.shape
        channels = None

    if curr_z >= target_z:
        # Take middle slices if current Z >= target
        start_idx = (curr_z - target_z) // 2
        if channels is not None:
            return data[:, :, start_idx:start_idx + target_z, :]
        else:
            return data[:, :, start_idx:start_idx + target_z]
    else:
        # Pad symmetrically
        pad_total = target_z - curr_z
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before

        if channels is not None:
            # (X, Y, Z, channels) format
            padded = np.pad(
                data,
                pad_width=((0, 0), (0, 0), (pad_before, pad_after), (0, 0)),
                mode='constant',
                constant_values=0
            )
        else:
            # (X, Y, Z) format
            padded = np.pad(
                data,
                pad_width=((0, 0), (0, 0), (pad_before, pad_after)),
                mode='constant',
                constant_values=0
            )
        return padded

def resize_xy_slice(slice_2d, target_x, target_y, is_label=False):
    """
    Resize a single 2D slice using PIL (memory efficient).

    Args:
        slice_2d: 2D array (X, Y)
        target_x, target_y: Target dimensions
        is_label: If True, use nearest neighbor. If False, use bilinear.

    Returns:
        Resized 2D array
    """
    # Convert to PIL Image
    img_pil = Image.fromarray((slice_2d * 255).astype(np.uint8))

    # Choose interpolation
    if is_label:
        img_pil = img_pil.resize((target_y, target_x), Image.NEAREST)
    else:
        img_pil = img_pil.resize((target_y, target_x), Image.BILINEAR)

    # Convert back to numpy
    resized = np.array(img_pil) / 255.0

    return resized

def preprocess_image(img_path, target_x, target_y, target_z):
    """
    Preprocess image: resize XY per slice, pad Z (centered).
    """
    img_nib = nib.load(img_path)
    img_data = img_nib.get_fdata()
    affine = img_nib.affine

    # Ensure 4D format (X, Y, Z, channels)
    if img_data.ndim == 3:
        img_data = np.expand_dims(img_data, axis=-1)

    x, y, z, channels = img_data.shape

    # Normalize to 0-1 range for processing
    img_min = img_data.min()
    img_max = img_data.max()
    if img_max > img_min:
        img_normalized = (img_data - img_min) / (img_max - img_min)
    else:
        img_normalized = img_data

    # Process each slice and channel
    resized_slices = []

    for ch in range(channels):
        ch_slices = []
        for z_idx in range(z):
            slice_2d = img_normalized[:, :, z_idx, ch]
            # Resize XY
            resized_slice = resize_xy_slice(slice_2d, target_x, target_y, is_label=False)
            ch_slices.append(resized_slice)

        # Stack Z slices for this channel
        ch_volume = np.stack(ch_slices, axis=2)  # (target_x, target_y, z)
        resized_slices.append(ch_volume)

    # Stack channels
    img_resized = np.stack(resized_slices, axis=-1)  # (target_x, target_y, z, channels)

    # Denormalize
    if img_max > img_min:
        img_resized = img_resized * (img_max - img_min) + img_min

    # Pad Z
    img_padded = pad_z_axis(img_resized, target_z)

    return img_padded.astype(np.float32), affine

def preprocess_label(label_path, target_x, target_y, target_z):
    """
    Preprocess label: resize XY per slice, pad Z (centered).
    Uses nearest neighbor to preserve class labels.
    """
    label_nib = nib.load(label_path)
    label_data = label_nib.get_fdata()
    affine = label_nib.affine

    # Ensure 4D format
    if label_data.ndim == 3:
        label_data = np.expand_dims(label_data, axis=-1)

    x, y, z, channels = label_data.shape

    # Normalize to 0-1 for PIL processing
    label_min = label_data.min()
    label_max = label_data.max()
    if label_max > label_min:
        label_normalized = (label_data - label_min) / (label_max - label_min)
    else:
        label_normalized = label_data

    # Process each slice and channel
    resized_slices = []

    for ch in range(channels):
        ch_slices = []
        for z_idx in range(z):
            slice_2d = label_normalized[:, :, z_idx, ch]
            # Resize XY with nearest neighbor
            resized_slice = resize_xy_slice(slice_2d, target_x, target_y, is_label=True)
            ch_slices.append(resized_slice)

        # Stack Z slices
        ch_volume = np.stack(ch_slices, axis=2)
        resized_slices.append(ch_volume)

    # Stack channels
    label_resized = np.stack(resized_slices, axis=-1)

    # Denormalize and round to nearest integer
    if label_max > label_min:
        label_resized = label_resized * (label_max - label_min) + label_min

    label_resized = np.round(label_resized).astype(np.uint8)

    # Pad Z
    label_padded = pad_z_axis(label_resized, target_z)

    return label_padded.astype(np.uint8), affine

def verify_preprocessing(img_array, label_array, pair_name):
    """Verify preprocessed image-label compatibility."""
    result = {
        'pair_name': pair_name,
        'valid': True,
        'issues': []
    }

    # Check shapes match
    if img_array.shape != label_array.shape:
        result['issues'].append(
            f'Shape mismatch: Image {img_array.shape} vs Label {label_array.shape}'
        )
        result['valid'] = False

    # Verify target dimensions
    if img_array.shape != (TARGET_X, TARGET_Y, TARGET_Z, 2):
        result['issues'].append(
            f'Image shape {img_array.shape} does not match target ({TARGET_X}, {TARGET_Y}, {TARGET_Z}, 2)'
        )
        result['valid'] = False

    # Check label values
    unique_labels = np.unique(label_array)
    if not all(label in [0, 1, 2, 3, 4] for label in unique_labels):
        invalid = [l for l in unique_labels if l not in [0, 1, 2, 3, 4]]
        result['issues'].append(f'Invalid label values: {invalid}')
        result['valid'] = False

    return result

# ======================
# Main Processing
# ======================

print("\n" + "="*70)
print("PROCESSING IMAGE-LABEL PAIRS")
print("="*70)

pairs = get_file_pairs()
print(f"\nFound {len(pairs)} image-label pairs to process\n")

preprocessing_results = []
failed_pairs = []

for idx, (img_path, label_path) in enumerate(pairs, 1):
    pair_name = img_path.stem

    try:
        # Preprocess
        img_array, img_affine = preprocess_image(img_path, TARGET_X, TARGET_Y, TARGET_Z)
        label_array, label_affine = preprocess_label(label_path, TARGET_X, TARGET_Y, TARGET_Z)

        # Verify
        verify_result = verify_preprocessing(img_array, label_array, pair_name)

        if verify_result['valid']:
            # Save
            img_nib_out = nib.Nifti1Image(img_array, img_affine)
            img_output_path = IMAGES_OUTPUT / f"{pair_name}.nii.gz"
            nib.save(img_nib_out, img_output_path)

            label_nib_out = nib.Nifti1Image(label_array, label_affine)
            label_output_path = LABELS_OUTPUT / f"{pair_name}.nii.gz"
            nib.save(label_nib_out, label_output_path)

            preprocessing_results.append({
                'pair_name': pair_name,
                'status': 'success'
            })
        else:
            preprocessing_results.append({
                'pair_name': pair_name,
                'status': 'verification_failed',
                'issues': verify_result['issues']
            })
            failed_pairs.append(pair_name)

        if idx % 50 == 0:
            print(f"  Processed {idx}/{len(pairs)} pairs")

    except Exception as e:
        print(f"  ERROR processing {pair_name}: {str(e)[:80]}")
        preprocessing_results.append({
            'pair_name': pair_name,
            'status': 'error',
            'error': str(e)
        })
        failed_pairs.append(pair_name)

print(f"\nProcessing complete!")

# ======================
# Results
# ======================
print("\n" + "="*70)
print("PREPROCESSING RESULTS SUMMARY")
print("="*70)

successful = [r for r in preprocessing_results if r['status'] == 'success']
print(f"\nSuccessfully preprocessed: {len(successful)}/{len(pairs)}")
print(f"Success rate: {(len(successful) / len(pairs)) * 100:.1f}%")
print(f"Target shape: ({TARGET_X}, {TARGET_Y}, {TARGET_Z}, 2)")

if failed_pairs:
    print(f"\nFailed pairs ({len(failed_pairs)}):")
    for pair_name in failed_pairs[:10]:
        print(f"  - {pair_name}")
    if len(failed_pairs) > 10:
        print(f"  ... and {len(failed_pairs) - 10} more")

# Save results
config_path = OUTPUT_DIR / 'preprocessing_config.json'
config = {
    'status': 'complete',
    'total_pairs': len(pairs),
    'successful_pairs': len(successful),
    'failed_pairs': len(failed_pairs),
    'success_rate': round((len(successful) / len(pairs)) * 100, 1),
    'target_shape': [TARGET_X, TARGET_Y, TARGET_Z, 2],
    'output_directories': {
        'images': str(IMAGES_OUTPUT),
        'labels': str(LABELS_OUTPUT)
    }
}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"\nConfiguration saved: {config_path}")

print("\n" + "="*70)
print("PREPROCESSING COMPLETE")
print("="*70)
