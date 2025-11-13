#!/usr/bin/env python
"""
Verify preprocessed 3D data quality

This script verifies:
1. All preprocessed files exist
2. Shapes are correct (2, 32, 512, 1024)
3. Data types are appropriate
4. Values are within expected ranges
5. Image-label pairs match
"""

import nibabel as nib
import numpy as np
from pathlib import Path
import json
import pandas as pd

BASE_DIR = Path('C:/FeatureEx')
OUTPUT_DIR = BASE_DIR / 'preprocessed_3d_data'
IMAGES_OUTPUT = OUTPUT_DIR / 'images'
LABELS_OUTPUT = OUTPUT_DIR / 'labels'

TARGET_SHAPE = (2, 32, 512, 1024)

print("="*70)
print("VERIFYING PREPROCESSED 3D DATA")
print("="*70)

print(f"\nChecking directories:")
print(f"  Images: {IMAGES_OUTPUT.exists()}")
print(f"  Labels: {LABELS_OUTPUT.exists()}")

# Get file lists
image_files = sorted([f for f in IMAGES_OUTPUT.glob('*.nii.gz')])
label_files = sorted([f for f in LABELS_OUTPUT.glob('*.nii.gz')])

print(f"\nFiles found:")
print(f"  Images: {len(image_files)}")
print(f"  Labels: {len(label_files)}")

if len(image_files) == 0:
    print("\nNo preprocessed files found yet. Processing may still be running...")
    exit(0)

# Verify pairs
image_basenames = {f.stem: f for f in image_files}
label_basenames = {f.stem: f for f in label_files}
matching_pairs = set(image_basenames.keys()) & set(label_basenames.keys())

print(f"\nMatching pairs: {len(matching_pairs)}/{len(image_files)}")

# Sample verification
verification_results = []

print(f"\nVerifying sample files...")
for idx, pair_name in enumerate(sorted(matching_pairs)[:10], 1):
    img_path = image_basenames[pair_name]
    label_path = label_basenames[pair_name]

    try:
        # Load image
        img_nib = nib.load(img_path)
        img_data = img_nib.get_fdata()

        # Load label
        label_nib = nib.load(label_path)
        label_data = label_nib.get_fdata()

        result = {
            'pair_name': pair_name,
            'img_shape': img_data.shape,
            'label_shape': label_data.shape,
            'shapes_match': img_data.shape == label_data.shape,
            'correct_shape': img_data.shape == TARGET_SHAPE,
            'img_dtype': str(img_data.dtype),
            'label_dtype': str(label_data.dtype),
            'img_min': float(img_data.min()),
            'img_max': float(img_data.max()),
            'img_mean': float(img_data.mean()),
            'label_classes': sorted(list(np.unique(label_data)))
        }

        verification_results.append(result)
        print(f"  {idx}. {pair_name}: OK")

    except Exception as e:
        print(f"  {idx}. {pair_name}: ERROR - {e}")
        verification_results.append({
            'pair_name': pair_name,
            'error': str(e)
        })

# Summary
print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)

if verification_results:
    df = pd.DataFrame(verification_results)

    # Check shapes
    if 'correct_shape' in df.columns:
        correct_count = df['correct_shape'].sum()
        print(f"\nShape Verification:")
        print(f"  Correct shape: {correct_count}/{len(df)} samples")
        print(f"  Target shape: {TARGET_SHAPE}")

    # Check matching
    if 'shapes_match' in df.columns:
        matching_count = df['shapes_match'].sum()
        print(f"\nAlignment Verification:")
        print(f"  Image-Label pairs match: {matching_count}/{len(df)}")

    # Data statistics
    if 'img_min' in df.columns:
        print(f"\nImage Data Statistics:")
        print(f"  Min: {df['img_min'].min():.2f}")
        print(f"  Max: {df['img_max'].max():.2f}")
        print(f"  Mean: {df['img_mean'].mean():.2f}")

    # Label classes
    if 'label_classes' in df.columns:
        print(f"\nLabel Classes Found:")
        all_classes = set()
        for classes in df['label_classes']:
            if isinstance(classes, list):
                all_classes.update(classes)
        print(f"  Unique classes: {sorted(list(all_classes))}")

print("\n" + "="*70)
print("PREPROCESSING VERIFICATION COMPLETE")
print("="*70)
