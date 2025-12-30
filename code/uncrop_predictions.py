"""
Uncrop predictions back to original image size for 3D Slicer visualization

This script:
1. Loads cropped predictions from nnUNet
2. Finds corresponding original images
3. Determines crop boundaries by analyzing the data
4. Creates full-size prediction masks matching original dimensions
5. Saves uncropped predictions that align with original images in 3D Slicer
"""

import os
import json
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path

# Paths
BASE_PATH = "/home/minhchau/anaconda3/envs/curvas/curvas"
PREDICTIONS_DIR = f"{BASE_PATH}/test_results/predictions"
ORIGINAL_DATA_DIR = f"{BASE_PATH}/Data"
CROPPED_LABELS_DIR = f"{BASE_PATH}/nnUNet_raw/Dataset001_PDAC/labelsTs"
OUTPUT_DIR = f"{BASE_PATH}/test_results/predictions_uncropped"
OUTPUT_IMAGES_DIR = f"{BASE_PATH}/test_results/images_original"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

# Load the mapping from preprocessing (PDAC_XXXX -> CURVASPDAC_XXXXX)
def get_case_mapping():
    """
    Create mapping from PDAC_XXXX to original CURVASPDAC_XXXXX names
    """
    # Read the preprocessing script to understand the mapping
    data_cases = sorted([d for d in os.listdir(ORIGINAL_DATA_DIR) 
                        if d.startswith('CURVASPDAC_') and 
                        os.path.isdir(os.path.join(ORIGINAL_DATA_DIR, d))])
    
    # Create mapping: PDAC_0001 -> CURVASPDAC_00003, etc.
    mapping = {}
    for idx, case_name in enumerate(data_cases, start=1):
        pdac_name = f"PDAC_{idx:04d}"
        mapping[pdac_name] = case_name
    
    return mapping

def find_crop_bounds_from_mask(original_mask_path, cropped_mask_path):
    """
    Find Z-axis crop boundaries by comparing original and cropped masks
    
    Args:
        original_mask_path: Path to original full-size mask
        cropped_mask_path: Path to cropped mask
    
    Returns:
        z_start, z_end: Indices where the crop was taken from
    """
    # Load masks
    orig_mask = sitk.ReadImage(original_mask_path)
    crop_mask = sitk.ReadImage(cropped_mask_path)
    
    orig_array = sitk.GetArrayFromImage(orig_mask)  # (Z, Y, X)
    crop_array = sitk.GetArrayFromImage(crop_mask)
    
    orig_size = orig_array.shape
    crop_size = crop_array.shape
    
    print(f"    Original size: {orig_size}")
    print(f"    Cropped size:  {crop_size}")
    
    # Find tumor center in original mask
    tumor_slices = np.where(np.any(orig_array > 0, axis=(1, 2)))[0]
    if len(tumor_slices) == 0:
        print(f"    ⚠️  No tumor found in original mask!")
        # Default: assume crop from middle
        z_center = orig_size[0] // 2
        z_start = max(0, z_center - crop_size[0] // 2)
        z_end = z_start + crop_size[0]
        return z_start, z_end
    
    tumor_z_min = tumor_slices[0]
    tumor_z_max = tumor_slices[-1]
    tumor_z_center = (tumor_z_min + tumor_z_max) // 2
    
    print(f"    Tumor Z range: {tumor_z_min}-{tumor_z_max} (center: {tumor_z_center})")
    
    # Crop was done ±40 slices around tumor, then possibly center-cropped
    # Calculate where the cropped region should start
    margin = 40
    ideal_z_start = max(0, tumor_z_min - margin)
    ideal_z_end = min(orig_size[0], tumor_z_max + margin + 1)
    ideal_size = ideal_z_end - ideal_z_start
    
    # If cropped size matches ideal size, use those bounds
    if crop_size[0] == ideal_size:
        z_start = ideal_z_start
        z_end = ideal_z_end
    else:
        # Otherwise, center the crop around tumor center
        z_start = max(0, tumor_z_center - crop_size[0] // 2)
        z_end = z_start + crop_size[0]
        
        # Adjust if out of bounds
        if z_end > orig_size[0]:
            z_end = orig_size[0]
            z_start = z_end - crop_size[0]
    
    print(f"    Detected crop bounds: Z[{z_start}:{z_end}]")
    
    return z_start, z_end

def uncrop_prediction(pred_path, original_image_path, cropped_mask_path, output_path, case_name):
    """
    Uncrop a prediction back to original image size
    
    Args:
        pred_path: Path to cropped prediction
        original_image_path: Path to original full-size image
        cropped_mask_path: Path to cropped ground truth (to find crop bounds)
        output_path: Where to save uncropped prediction
        case_name: Case identifier for logging
    """
    print(f"\n  Processing {case_name}...")
    
    # Load original image for size and metadata
    orig_img = sitk.ReadImage(original_image_path)
    orig_size = orig_img.GetSize()  # (X, Y, Z)
    orig_spacing = orig_img.GetSpacing()
    orig_origin = orig_img.GetOrigin()
    orig_direction = orig_img.GetDirection()
    
    # Load cropped prediction
    pred_img = sitk.ReadImage(pred_path)
    pred_array = sitk.GetArrayFromImage(pred_img)  # (Z, Y, X)
    
    # Find crop boundaries
    original_mask_path = original_image_path.replace('image.nii.gz', 'annotation_staple.nii.gz')
    z_start, z_end = find_crop_bounds_from_mask(original_mask_path, cropped_mask_path)
    
    # Create full-size array (zeros = background)
    full_array = np.zeros((orig_size[2], orig_size[1], orig_size[0]), dtype=pred_array.dtype)
    
    # Handle XY dimensions (might also be cropped 1024->512)
    pred_shape = pred_array.shape  # (Z, Y, X)
    
    # Check if XY was cropped
    if pred_shape[1] != orig_size[1] or pred_shape[2] != orig_size[0]:
        # XY was cropped (1024 -> 512), need to center it
        y_start = (orig_size[1] - pred_shape[1]) // 2
        x_start = (orig_size[0] - pred_shape[2]) // 2
        
        # Place prediction in the center of each slice
        full_array[z_start:z_end, 
                   y_start:y_start+pred_shape[1], 
                   x_start:x_start+pred_shape[2]] = pred_array
        
        print(f"    XY also cropped: placed at Y[{y_start}:{y_start+pred_shape[1]}], X[{x_start}:{x_start+pred_shape[2]}]")
    else:
        # Only Z was cropped
        full_array[z_start:z_end, :, :] = pred_array
    
    # Create new image with original metadata
    uncropped_img = sitk.GetImageFromArray(full_array)
    uncropped_img.SetSpacing(orig_spacing)
    uncropped_img.SetOrigin(orig_origin)
    uncropped_img.SetDirection(orig_direction)
    
    # Save
    sitk.WriteImage(uncropped_img, output_path)
    print(f"    ✅ Saved uncropped prediction: {output_path}")
    print(f"    Final size: {full_array.shape} (Z, Y, X)")

def main():
    print("="*80)
    print("Uncropping Predictions to Original Image Size")
    print("="*80)
    
    # Get case mapping
    print("\n📋 Creating case mapping...")
    case_mapping = get_case_mapping()
    print(f"   Found {len(case_mapping)} cases")
    
    # Get all predictions
    pred_files = sorted(Path(PREDICTIONS_DIR).glob("*.nii.gz"))
    pred_files = [f for f in pred_files if not f.name.endswith('_probs.nii.gz')]  # Skip probability files
    
    print(f"\n📂 Found {len(pred_files)} predictions to uncrop\n")
    
    uncropped_count = 0
    
    for pred_file in pred_files:
        # Get case ID: PDAC_0029.nii.gz -> PDAC_0029
        pred_name = pred_file.stem.replace('.nii', '')
        
        # Get original case name
        if pred_name not in case_mapping:
            print(f"⚠️  Skipping {pred_name}: not in mapping")
            continue
        
        original_case_name = case_mapping[pred_name]
        
        # Paths
        original_image_path = os.path.join(ORIGINAL_DATA_DIR, original_case_name, "image.nii.gz")
        cropped_mask_path = os.path.join(CROPPED_LABELS_DIR, f"{pred_name}.nii.gz")
        output_pred_path = os.path.join(OUTPUT_DIR, f"{original_case_name}_prediction.nii.gz")
        output_image_path = os.path.join(OUTPUT_IMAGES_DIR, f"{original_case_name}_image.nii.gz")
        
        # Check files exist
        if not os.path.exists(original_image_path):
            print(f"⚠️  Missing original image: {original_image_path}")
            continue
        
        if not os.path.exists(cropped_mask_path):
            print(f"⚠️  Missing cropped mask: {cropped_mask_path}")
            continue
        
        # Uncrop the prediction
        try:
            uncrop_prediction(
                str(pred_file),
                original_image_path,
                cropped_mask_path,
                output_pred_path,
                original_case_name
            )
            
            # Also copy the original image for convenience
            import shutil
            shutil.copy(original_image_path, output_image_path)
            print(f"    ✅ Copied original image: {output_image_path}")
            
            uncropped_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {pred_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("✅ Uncropping Complete!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Uncropped: {uncropped_count} predictions")
    print(f"\n📁 Output:")
    print(f"   Uncropped predictions: {OUTPUT_DIR}/")
    print(f"   Original images:       {OUTPUT_IMAGES_DIR}/")
    
    print(f"\n🔍 How to View in 3D Slicer:")
    print(f"   1. Open 3D Slicer")
    print(f"   2. File → Add Data → Select both:")
    print(f"      - {OUTPUT_IMAGES_DIR}/CURVASPDAC_XXXXX_image.nii.gz")
    print(f"      - {OUTPUT_DIR}/CURVASPDAC_XXXXX_prediction.nii.gz")
    print(f"   3. In Volumes module:")
    print(f"      - Background: image")
    print(f"      - Label: prediction")
    print(f"   4. The prediction should now align perfectly with the original image!")
    
    print(f"\n💡 The uncropped predictions are now the SAME SIZE and COORDINATE SYSTEM")
    print(f"   as your original images, so they will overlay correctly in any viewer.")

if __name__ == "__main__":
    main()
