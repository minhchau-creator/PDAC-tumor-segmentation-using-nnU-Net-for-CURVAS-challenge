"""
Convert CURVAS PDAC dataset to nnU-Net format with minimal preprocessing:
- Crop Z dimension around tumor region (±40 slices from tumor extent)
- Crop XY from 1024x1024 to 512x512 (center crop for 2 outlier cases)
- Rename files following nnU-Net convention
- Create dataset.json metadata file

nnU-Net will handle:
- Resampling/resizing automatically
- Intensity normalization
- Augmentation on-the-fly
"""

import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json
from tqdm import tqdm

# Paths
DATA_ROOT = "/home/minhchau/anaconda3/envs/curvas/curvas/Data"
OUTPUT_ROOT = "/home/minhchau/anaconda3/envs/curvas/curvas/nnUNet_raw"
DATASET_ID = "001"
DATASET_NAME = "PDAC"

# Output folders
IMAGES_TR = os.path.join(OUTPUT_ROOT, f"Dataset{DATASET_ID}_{DATASET_NAME}", "imagesTr")
LABELS_TR = os.path.join(OUTPUT_ROOT, f"Dataset{DATASET_ID}_{DATASET_NAME}", "labelsTr")

# Create output directories
os.makedirs(IMAGES_TR, exist_ok=True)
os.makedirs(LABELS_TR, exist_ok=True)

def get_tumor_z_bounds(mask_array):
    """
    Find the Z-slice range where tumor exists
    Returns: (z_min, z_max) indices
    """
    # Find slices with tumor (non-zero voxels)
    tumor_slices = np.where(np.any(mask_array > 0, axis=(0, 1)))[0]
    
    if len(tumor_slices) == 0:
        # No tumor found, return full Z range
        return 0, mask_array.shape[2] - 1
    
    z_min = tumor_slices[0]
    z_max = tumor_slices[-1]
    
    return z_min, z_max

def crop_z_around_tumor(image_array, mask_array, margin=40):
    """
    Crop Z dimension to tumor region ± margin slices
    
    Args:
        image_array: (H, W, D) image array
        mask_array: (H, W, D) mask array
        margin: number of slices to include before/after tumor
    
    Returns:
        cropped_image, cropped_mask, crop_info dict
    """
    z_min, z_max = get_tumor_z_bounds(mask_array)
    
    # Add margin
    z_start = max(0, z_min - margin)
    z_end = min(image_array.shape[2], z_max + margin + 1)
    
    # Crop both image and mask
    cropped_image = image_array[:, :, z_start:z_end]
    cropped_mask = mask_array[:, :, z_start:z_end]
    
    crop_info = {
        'original_z': image_array.shape[2],
        'tumor_z_min': int(z_min),
        'tumor_z_max': int(z_max),
        'cropped_z_start': int(z_start),
        'cropped_z_end': int(z_end),
        'cropped_z_size': int(z_end - z_start)
    }
    
    return cropped_image, cropped_mask, crop_info

def crop_xy_to_512(image_array, mask_array):
    """
    Crop XY from 1024x1024 to 512x512 (center crop)
    Only applies if image is 1024x1024
    """
    h, w = image_array.shape[:2]
    
    if h == 1024 and w == 1024:
        # Center crop to 512x512
        start_h = (h - 512) // 2
        start_w = (w - 512) // 2
        
        image_array = image_array[start_h:start_h+512, start_w:start_w+512, :]
        mask_array = mask_array[start_h:start_h+512, start_w:start_w+512, :]
    
    return image_array, mask_array

def process_case(case_folder, case_id):
    """
    Process one case: load, crop Z around tumor, crop XY if needed, save in nnU-Net format
    
    Args:
        case_folder: path to CURVASPDAC_XXXXX folder
        case_id: numeric ID for nnU-Net naming (e.g., 1, 2, 3...)
    
    Returns:
        crop_info dict or None if failed
    """
    image_path = os.path.join(case_folder, "image.nii.gz")
    mask_path = os.path.join(case_folder, "annotation_staple.nii.gz")
    
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print(f"⚠️  Skipping {case_folder}: missing files")
        return None
    
    # Load image and mask
    image_sitk = sitk.ReadImage(image_path)
    mask_sitk = sitk.ReadImage(mask_path)
    
    # Get arrays (SimpleITK uses (Z, Y, X) order, transpose to (X, Y, Z))
    image_array = sitk.GetArrayFromImage(image_sitk).transpose(2, 1, 0)  # (H, W, D)
    mask_array = sitk.GetArrayFromImage(mask_sitk).transpose(2, 1, 0)
    
    # Store original spacing and origin
    original_spacing = image_sitk.GetSpacing()
    original_origin = image_sitk.GetOrigin()
    original_direction = image_sitk.GetDirection()
    
    # Crop XY if 1024x1024
    image_array, mask_array = crop_xy_to_512(image_array, mask_array)
    
    # Crop Z around tumor
    image_array, mask_array, crop_info = crop_z_around_tumor(image_array, mask_array, margin=40)
    
    # Convert back to (Z, Y, X) for SimpleITK
    image_array = image_array.transpose(2, 1, 0)
    mask_array = mask_array.transpose(2, 1, 0)
    
    # Create new SimpleITK images
    image_out = sitk.GetImageFromArray(image_array)
    mask_out = sitk.GetImageFromArray(mask_array)
    
    # Preserve spacing, origin, direction
    image_out.SetSpacing(original_spacing)
    image_out.SetOrigin(original_origin)
    image_out.SetDirection(original_direction)
    
    mask_out.SetSpacing(original_spacing)
    mask_out.SetOrigin(original_origin)
    mask_out.SetDirection(original_direction)
    
    # nnU-Net naming convention: PDAC_0001_0000.nii.gz (modality 0000 for CT)
    case_str = f"{case_id:04d}"
    image_output_path = os.path.join(IMAGES_TR, f"{DATASET_NAME}_{case_str}_0000.nii.gz")
    mask_output_path = os.path.join(LABELS_TR, f"{DATASET_NAME}_{case_str}.nii.gz")
    
    # Save
    sitk.WriteImage(image_out, image_output_path)
    sitk.WriteImage(mask_out, mask_output_path)
    
    crop_info['case_name'] = os.path.basename(case_folder)
    crop_info['output_shape'] = image_array.shape  # (Z, Y, X)
    
    return crop_info

def create_dataset_json(num_cases, crop_logs):
    """
    Create dataset.json file required by nnU-Net
    """
    dataset_json = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "PDAC": 1
        },
        "numTraining": num_cases,
        "file_ending": ".nii.gz",
        "name": DATASET_NAME,
        "description": "CURVAS PDAC dataset - Pancreatic Ductal Adenocarcinoma segmentation",
        "reference": "CURVAS consortium",
        "preprocessing": "Z cropped around tumor ±40 slices, XY cropped to 512 if 1024",
        "crop_statistics": {
            "min_z": min([log['cropped_z_size'] for log in crop_logs]),
            "max_z": max([log['cropped_z_size'] for log in crop_logs]),
            "mean_z": np.mean([log['cropped_z_size'] for log in crop_logs]),
            "median_z": np.median([log['cropped_z_size'] for log in crop_logs])
        }
    }
    
    json_path = os.path.join(OUTPUT_ROOT, f"Dataset{DATASET_ID}_{DATASET_NAME}", "dataset.json")
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"\n✅ Created {json_path}")
    print(f"   Z statistics: min={dataset_json['crop_statistics']['min_z']}, "
          f"max={dataset_json['crop_statistics']['max_z']}, "
          f"mean={dataset_json['crop_statistics']['mean_z']:.1f}, "
          f"median={dataset_json['crop_statistics']['median_z']:.1f}")

def main():
    print("=" * 80)
    print("Converting CURVAS PDAC to nnU-Net format with Z cropping around tumor")
    print("=" * 80)
    
    # Find all case folders
    case_folders = sorted([
        os.path.join(DATA_ROOT, d) 
        for d in os.listdir(DATA_ROOT) 
        if d.startswith("CURVASPDAC_") and os.path.isdir(os.path.join(DATA_ROOT, d))
    ])
    
    print(f"\nFound {len(case_folders)} cases")
    print(f"Output: {OUTPUT_ROOT}/Dataset{DATASET_ID}_{DATASET_NAME}/")
    print(f"  - imagesTr/: {IMAGES_TR}")
    print(f"  - labelsTr/: {LABELS_TR}\n")
    
    crop_logs = []
    successful = 0
    
    for idx, case_folder in enumerate(tqdm(case_folders, desc="Processing cases")):
        case_id = idx + 1  # nnU-Net case IDs start from 1
        crop_info = process_case(case_folder, case_id)
        
        if crop_info is not None:
            crop_logs.append(crop_info)
            successful += 1
    
    print(f"\n✅ Successfully processed {successful}/{len(case_folders)} cases")
    
    # Create dataset.json
    if successful > 0:
        create_dataset_json(successful, crop_logs)
        
        # Print summary
        print("\n" + "=" * 80)
        print("Preprocessing Summary:")
        print("=" * 80)
        for log in crop_logs:
            print(f"{log['case_name']}: "
                  f"Z {log['original_z']} → {log['cropped_z_size']} slices "
                  f"(tumor at {log['tumor_z_min']}-{log['tumor_z_max']}, "
                  f"cropped {log['cropped_z_start']}-{log['cropped_z_end']}), "
                  f"shape {log['output_shape']}")
        
        print("\n" + "=" * 80)
        print("Next steps:")
        print("=" * 80)
        print(f"1. Set environment variables:")
        print(f"   export nnUNet_raw={OUTPUT_ROOT}")
        print(f"   export nnUNet_preprocessed=/home/minhchau/anaconda3/envs/curvas/curvas/nnUNet_preprocessed")
        print(f"   export nnUNet_results=/home/minhchau/anaconda3/envs/curvas/curvas/nnUNet_results")
        print(f"")
        print(f"2. Verify dataset integrity:")
        print(f"   nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity")
        print(f"")
        print(f"3. Run nnU-Net preprocessing and planning:")
        print(f"   nnUNetv2_plan_and_preprocess -d {DATASET_ID} -c 3d_fullres")
        print(f"")
        print(f"4. Train nnU-Net (fold 0):")
        print(f"   nnUNetv2_train {DATASET_ID} 3d_fullres 0")
        print("=" * 80)
    else:
        print("\n❌ No cases were successfully processed")

if __name__ == "__main__":
    main()
