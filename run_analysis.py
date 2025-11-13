#!/usr/bin/env python
"""
Execute 3D compatibility analysis notebook
"""
import json
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("3D MEDICAL IMAGE DATA ANALYSIS WITH COMPATIBILITY TESTING")
print("="*70)

# ========================
# Setup Paths
# ========================
BASE_DIR = Path('C:/FeatureEx')
IMAGES_DIR = BASE_DIR / 'imagesTr' / 'imagesTr'
LABELS_DIR = BASE_DIR / 'labelsTr' / 'labelsTr'

print(f"\nImages directory: {IMAGES_DIR}")
print(f"Labels directory: {LABELS_DIR}")
print(f"Images exist: {IMAGES_DIR.exists()}")
print(f"Labels exist: {LABELS_DIR.exists()}")

# ========================
# Get File Lists
# ========================
image_files = sorted([f for f in IMAGES_DIR.glob('*.nii*')])
label_files = sorted([f for f in LABELS_DIR.glob('*.nii*')])

print(f"\nFound {len(image_files)} image files")
print(f"Found {len(label_files)} label files")

# ========================
# Analyze All Images
# ========================
print(f"\nAnalyzing all {len(image_files)} image files...")

image_info = []
num_channels = None
depth = height = width = None

for idx, img_path in enumerate(image_files, 1):
    try:
        img_nib = nib.load(img_path)
        img_data = img_nib.get_fdata()

        info = {
            'filename': img_path.name,
            'shape': img_data.shape,
            'dtype': img_data.dtype,
            'min': img_data.min(),
            'max': img_data.max(),
            'mean': img_data.mean(),
            'size_mb': img_path.stat().st_size / (1024**2)
        }
        image_info.append(info)

        if idx % 10 == 0:
            print(f"  Processed {idx}/{len(image_files)} images")
    except Exception as e:
        print(f"  ERROR loading {img_path.name}: {e}")

print(f"Successfully analyzed {len(image_info)} images")

df_images = pd.DataFrame(image_info)

# Check consistency
print("\nImage Shape Consistency Check:")
print("="*70)

unique_shapes = df_images['shape'].unique()
print(f"Number of unique image shapes: {len(unique_shapes)}")
for shape in unique_shapes:
    count = (df_images['shape'] == shape).sum()
    print(f"  {shape}: {count} images")

most_common_shape = df_images['shape'].value_counts().index[0]
print(f"\nMost common image shape: {most_common_shape}")

if len(most_common_shape) == 3:
    depth, height, width = most_common_shape
    num_channels = 1
elif len(most_common_shape) == 4:
    num_channels, depth, height, width = most_common_shape

print(f"\nImage Configuration:")
print(f"  Channels: {num_channels}")
print(f"  Depth (Z): {depth}")
print(f"  Height (Y): {height}")
print(f"  Width (X): {width}")

# ========================
# Analyze All Labels
# ========================
print(f"\nAnalyzing all {len(label_files)} label files...")

label_info = []
all_unique_labels = set()

for idx, label_path in enumerate(label_files, 1):
    try:
        label_nib = nib.load(label_path)
        label_data = label_nib.get_fdata()

        unique_labels = np.unique(label_data)
        all_unique_labels.update(unique_labels)

        info = {
            'filename': label_path.name,
            'shape': label_data.shape,
            'dtype': label_data.dtype,
            'min': label_data.min(),
            'max': label_data.max(),
            'unique_labels': len(unique_labels),
            'size_mb': label_path.stat().st_size / (1024**2)
        }
        label_info.append(info)

        if idx % 10 == 0:
            print(f"  Processed {idx}/{len(label_files)} labels")
    except Exception as e:
        print(f"  ERROR loading {label_path.name}: {e}")

print(f"Successfully analyzed {len(label_info)} label files")

df_labels = pd.DataFrame(label_info)

print("\nLabel Shape Consistency Check:")
print("="*70)

unique_label_shapes = df_labels['shape'].unique()
print(f"Number of unique label shapes: {len(unique_label_shapes)}")
for shape in unique_label_shapes:
    count = (df_labels['shape'] == shape).sum()
    print(f"  {shape}: {count} labels")

print("\nLabel Classes Found:")
print("="*70)

all_unique_labels = sorted(list(all_unique_labels))
print(f"Unique labels across all files: {all_unique_labels}")
print(f"Number of classes: {len(all_unique_labels)}")

print(f"\nLabel Mapping:")
for i, label in enumerate(all_unique_labels):
    if label == 0:
        print(f"  {label} = Background")
    else:
        print(f"  {label} = Class {int(label)}")

# ========================
# Match Images and Labels
# ========================
print("\nImage-Label Matching:")
print("="*70)

print(f"Number of images: {len(image_files)}")
print(f"Number of labels: {len(label_files)}")

image_basenames = {f.stem: f for f in image_files}
label_basenames = {f.stem: f for f in label_files}

matching_pairs = set(image_basenames.keys()) & set(label_basenames.keys())
print(f"\nMatching image-label pairs: {len(matching_pairs)}/{len(image_files)}")

if len(matching_pairs) != len(image_files):
    print(f"\n⚠️  WARNING: Not all images have matching labels!")
    missing_in_labels = set(image_basenames.keys()) - set(label_basenames.keys())
    if missing_in_labels:
        print(f"   Images without labels: {len(missing_in_labels)}")
else:
    print(f"✓ All images have matching labels!")

# ========================
# COMPREHENSIVE COMPATIBILITY TEST
# ========================
print("\n" + "="*70)
print("IMAGE-LABEL PAIR COMPATIBILITY TEST")
print("="*70)

compatibility_results = []
issues_found = []

print(f"\nTesting {len(matching_pairs)} paired images and labels...\n")

for idx, pair_name in enumerate(sorted(matching_pairs), 1):
    try:
        # Load image
        img_path = image_basenames[pair_name]
        img_nib = nib.load(img_path)
        img_data = img_nib.get_fdata()

        # Load label
        label_path = label_basenames[pair_name]
        label_nib = nib.load(label_path)
        label_data = label_nib.get_fdata()

        # Initialize results dict
        result = {
            'pair_name': pair_name,
            'img_file': img_path.name,
            'label_file': label_path.name,
            'compatible': True,
            'issues': []
        }

        # Test 1: Shape compatibility
        if img_data.shape != label_data.shape:
            issue = f'Shape mismatch: Image {img_data.shape} vs Label {label_data.shape}'
            result['issues'].append(issue)
            result['compatible'] = False

        # Test 2: Data type compatibility
        img_dtype = str(img_data.dtype)
        label_dtype = str(label_data.dtype)
        result['img_dtype'] = img_dtype
        result['label_dtype'] = label_dtype

        # Test 3: Check for NaN or infinite values
        img_has_nan = np.isnan(img_data).any()
        img_has_inf = np.isinf(img_data).any()
        label_has_nan = np.isnan(label_data).any()
        label_has_inf = np.isinf(label_data).any()

        if img_has_nan:
            result['issues'].append(f'Image has {np.isnan(img_data).sum()} NaN voxels')
            result['compatible'] = False
        if img_has_inf:
            result['issues'].append(f'Image has {np.isinf(img_data).sum()} infinite voxels')
            result['compatible'] = False
        if label_has_nan:
            result['issues'].append(f'Label has {np.isnan(label_data).sum()} NaN voxels')
            result['compatible'] = False
        if label_has_inf:
            result['issues'].append(f'Label has {np.isinf(label_data).sum()} infinite voxels')
            result['compatible'] = False

        # Test 4: Label values
        label_values = np.unique(label_data)
        result['label_values'] = sorted([int(x) for x in label_values])

        # Test 5: Spatial extent check
        img_nonzero = np.count_nonzero(img_data)
        label_nonzero = np.count_nonzero(label_data)
        result['img_nonzero_voxels'] = int(img_nonzero)
        result['label_nonzero_voxels'] = int(label_nonzero)

        if label_nonzero == 0:
            result['issues'].append('Label is completely empty')
            result['compatible'] = False

        # Test 6: Affine matrix compatibility
        img_affine = img_nib.affine
        label_affine = label_nib.affine
        affine_match = np.allclose(img_affine, label_affine)
        result['affine_match'] = bool(affine_match)

        compatibility_results.append(result)

        if result['issues']:
            issues_found.append((pair_name, result['issues']))

        if idx % 10 == 0:
            print(f'  Tested {idx}/{len(matching_pairs)} pairs')
    except Exception as e:
        print(f'  ERROR testing {pair_name}: {e}')
        compatibility_results.append({
            'pair_name': pair_name,
            'compatible': False,
            'issues': [str(e)]
        })
        issues_found.append((pair_name, [str(e)]))

print(f'\nCompatibility testing complete!')
print(f'Total pairs tested: {len(compatibility_results)}')

# ========================
# COMPATIBILITY TEST RESULTS
# ========================
print("\n" + "="*70)
print("COMPATIBILITY TEST RESULTS")
print("="*70)

compatible_count = sum(1 for r in compatibility_results if r['compatible'])
incompatible_count = len(compatibility_results) - compatible_count

print(f"\nCompatible pairs: {compatible_count}/{len(compatibility_results)}")
print(f"Incompatible pairs: {incompatible_count}/{len(compatibility_results)}")

if issues_found:
    print(f"\n⚠️  ISSUES FOUND ({len(issues_found)} pairs with problems):")
    print("-"*70)
    for pair_name, issues in issues_found[:10]:
        print(f"\n{pair_name}:")
        for issue in issues:
            print(f"  - {issue}")
    if len(issues_found) > 10:
        print(f"\n... and {len(issues_found) - 10} more pairs with issues")
else:
    print(f"\n✅ ALL PAIRS ARE COMPATIBLE!")

df_compat = pd.DataFrame(compatibility_results)
print(f"\nCompatibility Summary:")
print(f"  Total tested: {len(df_compat)}")
print(f"  Compatible: {df_compat['compatible'].sum()}")
print(f"  Incompatible: {(~df_compat['compatible']).sum()}")
print(f"  Success rate: {(df_compat['compatible'].sum() / len(df_compat)) * 100:.1f}%")

# ========================
# DETAILED COMPATIBILITY REPORT
# ========================
print("\n" + "="*70)
print("DETAILED COMPATIBILITY REPORT")
print("="*70)

all_issues = []
for pair_name, issues in issues_found:
    all_issues.extend(issues)

if all_issues:
    issue_counts = Counter(all_issues)
    print(f"\nIssue Summary:")
    for issue, count in issue_counts.most_common():
        print(f"  - {count}x: {issue[:70]}")

print(f"\nData Type Consistency:")
img_dtypes = df_compat['img_dtype'].value_counts()
label_dtypes = df_compat['label_dtype'].value_counts()
print(f"  Image types: {dict(img_dtypes)}")
print(f"  Label types: {dict(label_dtypes)}")

if 'affine_match' in df_compat.columns:
    affine_counts = df_compat['affine_match'].value_counts()
    print(f"\nAffine Matrix Compatibility:")
    print(f"  Matching: {affine_counts.get(True, 0)}")
    print(f"  Not matching: {affine_counts.get(False, 0)}")

print(f"\n" + "="*70)
if incompatible_count == 0:
    print(f"✅ DATASET IS READY FOR PROCESSING")
    print(f"   All {len(df_compat)} image-label pairs are compatible")
else:
    print(f"⚠️  DATASET REQUIRES PREPROCESSING")
    print(f"   {incompatible_count} pairs have compatibility issues")
    print(f"   Recommend fixing before pipeline training")
print(f"="*70)

# ========================
# FINAL CONFIGURATION SUMMARY
# ========================
print("\n" + "="*70)
print("3D PIPELINE CONFIGURATION SUMMARY")
print("="*70 + "\n")

config = {
    'dataset': {
        'num_images': len(image_files),
        'num_labels': len(label_files),
        'matching_pairs': len(matching_pairs),
        'compatible_pairs': compatible_count,
        'incompatible_pairs': incompatible_count,
    },
    'image_dimensions': {
        'channels': num_channels,
        'depth': depth,
        'height': height,
        'width': width,
    },
    'labels': {
        'unique_labels': list(all_unique_labels),
        'num_classes': len(all_unique_labels),
        'has_background': 0 in all_unique_labels,
    },
    'data_quality': {
        'all_compatible': incompatible_count == 0,
        'compatibility_rate': round((compatible_count / len(compatibility_results)) * 100, 1),
    }
}

for section, values in config.items():
    print(f"{section.upper()}:")
    for key, value in values.items():
        print(f"  {key}: {value}")
    print()

# Save configuration
config_path = BASE_DIR / '3d_dataset_config.json'
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"\nConfiguration saved to: {config_path}")
print("\n" + "="*70)
print("Analysis complete!")
print("="*70)
