"""
Direct training script for nnU-Net - no prompts, just run
"""

import os
import subprocess

# Setup environment
BASE_PATH = "/home/minhchau/anaconda3/envs/curvas/curvas"
os.environ['nnUNet_raw'] = f"{BASE_PATH}/nnUNet_raw"
os.environ['nnUNet_preprocessed'] = f"{BASE_PATH}/nnUNet_preprocessed"
os.environ['nnUNet_results'] = f"{BASE_PATH}/nnUNet_results"

os.makedirs(os.environ['nnUNet_preprocessed'], exist_ok=True)
os.makedirs(os.environ['nnUNet_results'], exist_ok=True)

print("Environment variables:")
print(f"  nnUNet_raw: {os.environ['nnUNet_raw']}")
print(f"  nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
print(f"  nnUNet_results: {os.environ['nnUNet_results']}")
print("")

DATASET_ID = "001"

# Step 1: Verify dataset
print("=" * 80)
print("STEP 1: Verifying dataset integrity...")
print("=" * 80)
subprocess.run(
    ["nnUNetv2_plan_and_preprocess", "-d", DATASET_ID, "--verify_dataset_integrity"],
    check=False
)

# Step 2: Preprocessing and planning
print("\n" + "=" * 80)
print("STEP 2: Running preprocessing and planning...")
print("=" * 80)
subprocess.run(
    ["nnUNetv2_plan_and_preprocess", "-d", DATASET_ID, "-c", "3d_fullres"],
    check=True
)

# Step 3: Training
print("\n" + "=" * 80)
print("STEP 3: Starting training (fold 0)...")
print("=" * 80)
subprocess.run(
    ["nnUNetv2_train", DATASET_ID, "3d_fullres", "0"],
    check=True
)

print("\n🎉 All steps completed!")
