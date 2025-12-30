"""
Test nnU-Net model on Task7_PDAC data and organize results

This script works with Task7_PDAC format (original 512x512 images):
1. Runs inference on test images using trained checkpoint
2. Uncrops predictions back to original image size
3. Organizes output in the format:
   {output_dir}/PDAC_{id}/
   ├── PDAC_{id}_img.nii.gz (original image)
   ├── PDAC_{id}_groundtruth.nii.gz (ground truth)
   └── PDAC_{id}_predict.nii.gz (prediction at original size)

Usage:
    python test_model_task7.py --input /path/to/images --labels /path/to/labels --output /path/to/output
    python test_model_task7.py  (will use default paths)
"""

import os
import sys
import argparse
import subprocess
import shutil
import numpy as np
import SimpleITK as sitk

# Setup nnU-Net environment variables
BASE_PATH = "/home/minhchau/anaconda3/envs/curvas/curvas"
os.environ['nnUNet_raw'] = f"{BASE_PATH}/nnUNet_raw"
os.environ['nnUNet_preprocessed'] = f"{BASE_PATH}/nnUNet_preprocessed"
os.environ['nnUNet_results'] = f"{BASE_PATH}/nnUNet_results"

# Default paths for Task7_PDAC
DEFAULT_INPUT = f"{BASE_PATH}/nnUNet_raw/Task7_PDAC/imagesTr"
DEFAULT_LABELS = f"{BASE_PATH}/nnUNet_raw/Task7_PDAC/labelsTr"
DEFAULT_OUTPUT = f"{BASE_PATH}/test_results/task7_results"

# Model configuration
DATASET_ID = "001"  # or "7" if using Task7
CHECKPOINT_TYPE = "checkpoint_best.pth"
NNUNET_CMD = "/home/minhchau/anaconda3/envs/curvas/bin/nnUNetv2_predict"

# Global variables
TEST_INPUT = None
TEST_LABELS = None
OUTPUT_DIR = None
TEMP_DIR = None
PREDICTIONS_DIR = None

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test nnU-Net model on Task7_PDAC data')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT,
                        help=f'Input images folder (default: {DEFAULT_INPUT})')
    parser.add_argument('--labels', type=str, default=DEFAULT_LABELS,
                        help=f'Ground truth labels folder (default: {DEFAULT_LABELS})')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help=f'Output directory (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_TYPE,
                        help=f'Checkpoint to use (default: {CHECKPOINT_TYPE})')
    parser.add_argument('--dataset-id', type=str, default=DATASET_ID,
                        help=f'Dataset ID (default: {DATASET_ID})')
    return parser.parse_args()

def setup_paths(args):
    """Setup global path variables"""
    global TEST_INPUT, TEST_LABELS, OUTPUT_DIR, TEMP_DIR, PREDICTIONS_DIR, CHECKPOINT_TYPE, DATASET_ID

    TEST_INPUT = args.input
    TEST_LABELS = args.labels
    OUTPUT_DIR = args.output
    CHECKPOINT_TYPE = args.checkpoint
    DATASET_ID = args.dataset_id

    TEMP_DIR = f"{BASE_PATH}/temp_inference_task7"
    PREDICTIONS_DIR = f"{TEMP_DIR}/predictions"

def run_inference():
    """Run nnU-Net inference"""
    print("\n" + "="*80)
    print("Step 1: Running nnU-Net Inference")
    print("="*80)

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    cmd = [
        NNUNET_CMD,
        "-i", TEST_INPUT,
        "-o", PREDICTIONS_DIR,
        "-d", DATASET_ID,
        "-c", "3d_fullres",
        "-f", "0",
        "-chk", CHECKPOINT_TYPE,
        "--disable_progress_bar"  # Reduce overhead
    ]

    print(f"Command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Inference completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Inference failed with error code: {e.returncode}")
        return False

def find_crop_bounds(original_img_array, cropped_shape):
    """Find crop boundaries for uncropping prediction"""
    orig_shape = original_img_array.shape

    z_start = max(0, (orig_shape[0] - cropped_shape[0]) // 2)
    z_end = z_start + cropped_shape[0]

    y_start = max(0, (orig_shape[1] - cropped_shape[1]) // 2)
    y_end = y_start + cropped_shape[1]

    x_start = max(0, (orig_shape[2] - cropped_shape[2]) // 2)
    x_end = x_start + cropped_shape[2]

    return z_start, z_end, y_start, y_end, x_start, x_end

def uncrop_prediction(pred_path, original_img_path):
    """Uncrop prediction back to original image size"""
    # Load images
    orig_img = sitk.ReadImage(original_img_path)
    pred_img = sitk.ReadImage(pred_path)

    # Get arrays
    orig_array = sitk.GetArrayFromImage(orig_img)
    pred_array = sitk.GetArrayFromImage(pred_img)

    # Get metadata
    orig_spacing = orig_img.GetSpacing()
    orig_origin = orig_img.GetOrigin()
    orig_direction = orig_img.GetDirection()

    # If sizes match, no need to uncrop
    if orig_array.shape == pred_array.shape:
        uncropped_img = pred_img
        uncropped_img.SetSpacing(orig_spacing)
        uncropped_img.SetOrigin(orig_origin)
        uncropped_img.SetDirection(orig_direction)
        return uncropped_img

    # Create full-size array
    full_array = np.zeros(orig_array.shape, dtype=pred_array.dtype)

    # Find crop bounds
    z_start, z_end, y_start, y_end, x_start, x_end = find_crop_bounds(orig_array, pred_array.shape)

    # Place cropped prediction in full array
    full_array[z_start:z_end, y_start:y_end, x_start:x_end] = pred_array

    # Create image with original metadata
    uncropped_img = sitk.GetImageFromArray(full_array)
    uncropped_img.SetSpacing(orig_spacing)
    uncropped_img.SetOrigin(orig_origin)
    uncropped_img.SetDirection(orig_direction)

    return uncropped_img

def organize_results():
    """
    Organize results into structured format

    Input format: PDAC_XXXX_0000.nii.gz (images), PDAC_XXXX.nii.gz (labels/predictions)
    Output: {OUTPUT_DIR}/PDAC_{id}/
            ├── PDAC_{id}_img.nii.gz
            ├── PDAC_{id}_groundtruth.nii.gz
            └── PDAC_{id}_predict.nii.gz
    """
    print("\n" + "="*80)
    print("Step 2: Organizing Results and Uncropping Predictions")
    print("="*80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all image files (format: PDAC_XXXX_0000.nii.gz)
    from pathlib import Path
    image_files = sorted(Path(TEST_INPUT).glob("*_0000.nii.gz"))
    print(f"\nFound {len(image_files)} test images")

    # Check what predictions were actually created
    if os.path.exists(PREDICTIONS_DIR):
        pred_files = sorted([f for f in os.listdir(PREDICTIONS_DIR) if f.endswith('.nii.gz')])
        print(f"Found {len(pred_files)} prediction files")
        if len(pred_files) < len(image_files):
            print(f"⚠️  Warning: {len(image_files) - len(pred_files)} predictions are missing!")
    else:
        print(f"❌ Predictions directory not found: {PREDICTIONS_DIR}")
        return 0, len(image_files)

    processed_count = 0
    failed_cases = []

    for img_file in image_files:
        # Extract case ID: PDAC_0001_0000.nii.gz -> PDAC_0001
        case_id = img_file.stem.replace('_0000', '').replace('.nii', '')

        print(f"\n  Processing {case_id}...")

        # Create output folder for this case
        case_output_dir = os.path.join(OUTPUT_DIR, case_id)
        os.makedirs(case_output_dir, exist_ok=True)

        # Define paths
        output_img_path = os.path.join(case_output_dir, f"{case_id}_img.nii.gz")
        output_gt_path = os.path.join(case_output_dir, f"{case_id}_groundtruth.nii.gz")
        output_pred_path = os.path.join(case_output_dir, f"{case_id}_predict.nii.gz")

        # Copy original image
        shutil.copy(str(img_file), output_img_path)
        print(f"    ✅ Copied image: {case_id}_img.nii.gz")

        # Copy ground truth if available
        if TEST_LABELS and os.path.exists(TEST_LABELS):
            gt_file = os.path.join(TEST_LABELS, f"{case_id}.nii.gz")
            if os.path.exists(gt_file):
                shutil.copy(gt_file, output_gt_path)
                print(f"    ✅ Copied ground truth: {case_id}_groundtruth.nii.gz")
            else:
                print(f"    ⚠️  Ground truth not found: {gt_file}")
        else:
            print(f"    ⚠️  No ground truth directory")

        # Uncrop and save prediction
        pred_file = os.path.join(PREDICTIONS_DIR, f"{case_id}.nii.gz")
        if os.path.exists(pred_file):
            try:
                # Uncrop prediction to match original image size
                uncropped_pred = uncrop_prediction(pred_file, str(img_file))
                sitk.WriteImage(uncropped_pred, output_pred_path)

                # Verify sizes match
                orig_img = sitk.ReadImage(str(img_file))
                pred_img = sitk.ReadImage(output_pred_path)

                orig_size = orig_img.GetSize()
                pred_size = pred_img.GetSize()

                if orig_size == pred_size:
                    print(f"    ✅ Saved prediction (uncropped): {case_id}_predict.nii.gz")
                    print(f"       Size verified: {pred_size}")

                    # Check if prediction has any non-zero values
                    pred_array = sitk.GetArrayFromImage(pred_img)
                    non_zero = np.sum(pred_array > 0)
                    print(f"       Predicted tumor voxels: {non_zero}")
                else:
                    print(f"    ⚠️  Size mismatch! Original: {orig_size}, Prediction: {pred_size}")

                processed_count += 1

            except Exception as e:
                print(f"    ❌ Error processing prediction: {e}")
                import traceback
                traceback.print_exc()
                failed_cases.append(case_id)
        else:
            print(f"    ❌ Prediction file not found: {pred_file}")
            failed_cases.append(case_id)

    if failed_cases:
        print(f"\n⚠️  Failed cases ({len(failed_cases)}): {', '.join(failed_cases)}")

    return processed_count, len(image_files)

def print_summary(processed_count, total_count):
    """Print summary and instructions"""
    print("\n" + "="*80)
    print("✅ Testing Complete!")
    print("="*80)

    print(f"\n📊 Results Summary:")
    print(f"   - Total images: {total_count}")
    print(f"   - Successfully processed: {processed_count}")

    print(f"\n📁 Output Structure:")
    print(f"   {OUTPUT_DIR}/")

    # Show example case structure
    if os.path.exists(OUTPUT_DIR):
        cases = sorted([d for d in os.listdir(OUTPUT_DIR)
                       if os.path.isdir(os.path.join(OUTPUT_DIR, d))])
        if cases:
            example_case = cases[0]
            example_dir = os.path.join(OUTPUT_DIR, example_case)
            if os.path.exists(example_dir):
                files = sorted(os.listdir(example_dir))

                print(f"   ├── {example_case}/")
                for f in files:
                    print(f"   │   ├── {f}")

                if len(cases) > 1:
                    print(f"   ├── ... ({len(cases) - 1} more cases)")

    print(f"\n🔍 How to View in 3D Slicer:")
    print(f"   For each case folder (e.g., PDAC_0001/):")
    print(f"   1. Open 3D Slicer")
    print(f"   2. File → Add Data → Select all 3 files:")
    print(f"      - PDAC_XXXX_img.nii.gz (CT image)")
    print(f"      - PDAC_XXXX_groundtruth.nii.gz (ground truth)")
    print(f"      - PDAC_XXXX_predict.nii.gz (prediction)")
    print(f"   3. All files are now the same size and will align perfectly!")

def cleanup_temp():
    """Clean up temporary directory"""
    if os.path.exists(TEMP_DIR):
        print(f"\n🧹 Cleaning up temporary files: {TEMP_DIR}")
        shutil.rmtree(TEMP_DIR)

def main():
    # Parse arguments
    args = parse_args()
    setup_paths(args)

    print("="*80)
    print("nnU-Net Model Testing on Task7_PDAC Data")
    print("="*80)
    print(f"Dataset ID: {DATASET_ID}")
    print(f"Checkpoint: {CHECKPOINT_TYPE}")
    print(f"Input images: {TEST_INPUT}")
    print(f"Input labels: {TEST_LABELS}")
    print(f"Output: {OUTPUT_DIR}")
    print("")

    # Verify input directory exists
    if not os.path.exists(TEST_INPUT):
        print(f"❌ Error: Input directory does not exist: {TEST_INPUT}")
        sys.exit(1)

    try:
        # Step 1: Run inference
        if not run_inference():
            print("\n❌ Inference failed. Please check the error messages above.")
            cleanup_temp()
            return

        # Step 2: Organize results and uncrop predictions
        processed_count, total_count = organize_results()

        # Step 3: Print summary
        print_summary(processed_count, total_count)

        # Step 4: Cleanup temporary files
        print(f"\n⚠️  Keeping temp directory for inspection: {TEMP_DIR}")
        print(f"    To clean up manually, run: rm -rf {TEMP_DIR}")
        # cleanup_temp()  # Disabled - keep files for debugging

    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user (Ctrl+C)")
        print(f"⚠️  Temp directory preserved: {TEMP_DIR}")
        # cleanup_temp()
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"\n⚠️  Temp directory preserved for debugging: {TEMP_DIR}")
        # cleanup_temp()
        sys.exit(1)

if __name__ == "__main__":
    main()
