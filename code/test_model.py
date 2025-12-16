"""
Test nnU-Net model and prepare results for 3D Slicer visualization
This script:
1. Runs inference on test images using trained checkpoint
2. Saves predictions as NIfTI masks
3. Copies original images for easy viewing in 3D Slicer
"""

import os
import subprocess
import shutil
from pathlib import Path

# Setup nnU-Net environment variables
BASE_PATH = "/home/minhchau/anaconda3/envs/curvas/curvas"
os.environ['nnUNet_raw'] = f"{BASE_PATH}/nnUNet_raw"
os.environ['nnUNet_preprocessed'] = f"{BASE_PATH}/nnUNet_preprocessed"
os.environ['nnUNet_results'] = f"{BASE_PATH}/nnUNet_results"

# Paths
DATASET_ID = "001"
TEST_INPUT = f"{BASE_PATH}/nnUNet_raw/Dataset{DATASET_ID}_PDAC/imagesTs"
TEST_LABELS = f"{BASE_PATH}/nnUNet_raw/Dataset{DATASET_ID}_PDAC/labelsTs"
OUTPUT_DIR = f"{BASE_PATH}/test_results"
PREDICTIONS_DIR = f"{OUTPUT_DIR}/predictions"
IMAGES_DIR = f"{OUTPUT_DIR}/images"
LABELS_DIR = f"{OUTPUT_DIR}/ground_truth"

# Model configuration
CHECKPOINT_TYPE = "checkpoint_best"  # or "checkpoint_final" or "checkpoint_latest"

def setup_output_dirs():
    """Create output directories"""
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)
    print(f"✅ Created output directories:")
    print(f"   - Predictions: {PREDICTIONS_DIR}")
    print(f"   - Images: {IMAGES_DIR}")
    print(f"   - Ground truth: {LABELS_DIR}")

def run_inference():
    """Run nnU-Net inference"""
    print("\n" + "="*80)
    print("Running nnU-Net Inference")
    print("="*80)
    
    cmd = [
        "nnUNetv2_predict",
        "-i", TEST_INPUT,
        "-o", PREDICTIONS_DIR,
        "-d", DATASET_ID,
        "-c", "3d_fullres",
        "-f", "0",
        "-chk", CHECKPOINT_TYPE,
        "--save_probabilities"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("")
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print("\n✅ Inference completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Inference failed with error code: {e.returncode}")
        return False

def copy_images_and_labels():
    """Copy original images and ground truth labels for visualization"""
    print("\n" + "="*80)
    print("Copying Images and Labels")
    print("="*80)
    
    # Copy images
    if os.path.exists(TEST_INPUT):
        image_files = list(Path(TEST_INPUT).glob("*.nii.gz"))
        print(f"\nCopying {len(image_files)} test images...")
        for img_file in image_files:
            # Remove _0000 suffix for cleaner naming
            dest_name = img_file.name.replace("_0000.nii.gz", ".nii.gz")
            shutil.copy(img_file, os.path.join(IMAGES_DIR, dest_name))
        print(f"✅ Copied {len(image_files)} images to {IMAGES_DIR}")
    
    # Copy ground truth labels if they exist
    if os.path.exists(TEST_LABELS):
        label_files = list(Path(TEST_LABELS).glob("*.nii.gz"))
        if label_files:
            print(f"\nCopying {len(label_files)} ground truth labels...")
            for label_file in label_files:
                shutil.copy(label_file, LABELS_DIR)
            print(f"✅ Copied {len(label_files)} labels to {LABELS_DIR}")
    else:
        print("\n⚠️  No ground truth labels found (this is normal for test set without annotations)")

def evaluate_results():
    """Evaluate predictions if ground truth is available"""
    print("\n" + "="*80)
    print("Evaluating Predictions")
    print("="*80)
    
    if not os.path.exists(TEST_LABELS) or not list(Path(TEST_LABELS).glob("*.nii.gz")):
        print("⚠️  No ground truth labels available, skipping evaluation")
        return
    
    print("\nRunning evaluation (Dice, Hausdorff Distance, Average Surface Distance)...")
    
    cmd = [
        "nnUNetv2_evaluate_folder",
        TEST_LABELS,
        PREDICTIONS_DIR,
        "-l", "1"  # Label 1 = tumor
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Evaluation completed!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Evaluation failed: {e.returncode}")

def print_summary():
    """Print summary and instructions"""
    print("\n" + "="*80)
    print("🎉 Testing Complete!")
    print("="*80)
    
    # Count files
    pred_files = list(Path(PREDICTIONS_DIR).glob("*.nii.gz"))
    img_files = list(Path(IMAGES_DIR).glob("*.nii.gz"))
    
    print(f"\n📊 Results Summary:")
    print(f"   - Test images: {len(img_files)}")
    print(f"   - Predictions: {len(pred_files)}")
    
    print(f"\n📁 Output Structure:")
    print(f"   {OUTPUT_DIR}/")
    print(f"   ├── images/          (original CT images)")
    print(f"   ├── predictions/     (predicted tumor masks)")
    print(f"   └── ground_truth/    (ground truth masks if available)")
    
    print(f"\n🔍 How to View in 3D Slicer:")
    print(f"   1. Open 3D Slicer")
    print(f"   2. Load image: File → Add Data → {IMAGES_DIR}/CURVASPDAC_XXXXX.nii.gz")
    print(f"   3. Load prediction: File → Add Data → {PREDICTIONS_DIR}/CURVASPDAC_XXXXX.nii.gz")
    print(f"   4. In Volumes module:")
    print(f"      - Set CT image as 'Background'")
    print(f"      - Set prediction as 'Label Map'")
    print(f"   5. Adjust window/level and opacity to visualize overlay")
    
    print(f"\n💡 Quick Slicer Command:")
    print(f"   Slicer --python-code \"")
    print(f"   slicer.util.loadVolume('{IMAGES_DIR}/CURVASPDAC_00003.nii.gz')")
    print(f"   slicer.util.loadLabelVolume('{PREDICTIONS_DIR}/CURVASPDAC_00003.nii.gz')\"")

def main():
    print("="*80)
    print("nnU-Net Model Testing & Visualization Preparation")
    print("="*80)
    print(f"Dataset: Dataset{DATASET_ID}_PDAC")
    print(f"Checkpoint: {CHECKPOINT_TYPE}")
    print(f"Test images: {TEST_INPUT}")
    print("")
    
    # Step 1: Setup
    setup_output_dirs()
    
    # Step 2: Run inference
    if not run_inference():
        print("\n❌ Inference failed. Please check the error messages above.")
        return
    
    # Step 3: Copy images and labels
    copy_images_and_labels()
    
    # Step 4: Evaluate if possible
    evaluate_results()
    
    # Step 5: Print summary
    print_summary()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
