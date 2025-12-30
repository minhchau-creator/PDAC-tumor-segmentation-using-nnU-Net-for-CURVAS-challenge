"""
Prepare test set from validation cases for nnU-Net testing
This script copies validation cases to the test set directory
"""

import os
import json
import shutil
from pathlib import Path

# Setup paths
BASE_PATH = "/home/minhchau/anaconda3/envs/curvas/curvas"
DATASET_ID = "001"
RAW_DATA_DIR = f"{BASE_PATH}/nnUNet_raw/Dataset{DATASET_ID}_PDAC"

# Load split from nnUNet preprocessed directory (fold 0)
with open(f"{BASE_PATH}/nnUNet_preprocessed/Dataset{DATASET_ID}_PDAC/splits_final.json", 'r') as f:
    splits = json.load(f)
    split = splits[0]  # Use fold 0

def prepare_test_set():
    """Prepare test set from validation cases"""
    
    # Create test directories
    test_images_dir = f"{RAW_DATA_DIR}/imagesTs"
    test_labels_dir = f"{RAW_DATA_DIR}/labelsTs"
    
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)
    
    print("="*80)
    print("Preparing Test Set from Validation Cases")
    print("="*80)
    
    train_dir = f"{RAW_DATA_DIR}/imagesTr"
    label_dir = f"{RAW_DATA_DIR}/labelsTr"
    
    val_cases = split['val']
    print(f"\nValidation cases to use as test set: {len(val_cases)}")
    print(f"Cases: {val_cases}\n")
    
    copied_images = 0
    copied_labels = 0
    
    for case in val_cases:
        # Copy image (with _0000 suffix)
        src_image = f"{train_dir}/{case}_0000.nii.gz"
        dst_image = f"{test_images_dir}/{case}_0000.nii.gz"
        
        if os.path.exists(src_image):
            shutil.copy(src_image, dst_image)
            copied_images += 1
            print(f"✅ Copied image: {case}_0000.nii.gz")
        else:
            print(f"❌ Missing image: {src_image}")
        
        # Copy label
        src_label = f"{label_dir}/{case}.nii.gz"
        dst_label = f"{test_labels_dir}/{case}.nii.gz"
        
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
            copied_labels += 1
            print(f"✅ Copied label: {case}.nii.gz")
        else:
            print(f"❌ Missing label: {src_label}")
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"✅ Copied {copied_images} test images to {test_images_dir}")
    print(f"✅ Copied {copied_labels} test labels to {test_labels_dir}")
    print(f"\nTest set ready for inference!")

if __name__ == "__main__":
    prepare_test_set()
