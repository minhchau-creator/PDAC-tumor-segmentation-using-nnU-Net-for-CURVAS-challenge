"""
Train nnU-Net with integrated monitoring
Runs training in background and shows progress in real-time
"""

import os
import subprocess
import threading
import time
import sys

BASE_PATH = "/home/minhchau/anaconda3/envs/curvas/curvas"
os.environ['nnUNet_raw'] = f"{BASE_PATH}/nnUNet_raw"
os.environ['nnUNet_preprocessed'] = f"{BASE_PATH}/nnUNet_preprocessed"
os.environ['nnUNet_results'] = f"{BASE_PATH}/nnUNet_results"

DATASET_ID = "001"
RESULTS_PATH = f"{BASE_PATH}/nnUNet_results/Dataset001_PDAC/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0"

def read_training_log_tail():
    """Read and display last lines of training log"""
    log_file = os.path.join(RESULTS_PATH, "training_log.txt")
    
    if not os.path.exists(log_file):
        return []
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        return lines[-10:]  # Last 10 lines
    except:
        return []

def monitor_progress():
    """Monitor training progress in background thread"""
    print("\n" + "=" * 80)
    print("Training Monitor Started")
    print("=" * 80)
    
    last_size = 0
    while True:
        time.sleep(5)  # Check every 5 seconds
        
        log_file = os.path.join(RESULTS_PATH, "training_log.txt")
        if os.path.exists(log_file):
            current_size = os.path.getsize(log_file)
            
            # If log file grew, show new content
            if current_size > last_size:
                lines = read_training_log_tail()
                
                # Extract key info from last lines
                for line in lines[-3:]:
                    if "Epoch" in line or "val_loss" in line or "train_loss" in line:
                        print(line.strip())
                
                last_size = current_size
        
        # Check if training is complete
        checkpoint_final = os.path.join(RESULTS_PATH, "checkpoint_final.pth")
        if os.path.exists(checkpoint_final):
            print("\n" + "=" * 80)
            print("🎉 Training Complete!")
            print("=" * 80)
            break

def train_nnunet():
    """Start nnU-Net training"""
    print("=" * 80)
    print("Starting nnU-Net Training with Live Monitoring")
    print("=" * 80)
    print(f"Dataset: 001 (PDAC)")
    print(f"Configuration: 3d_fullres")
    print(f"Fold: 0")
    print(f"Results: {RESULTS_PATH}")
    print("=" * 80)
    
    cmd = ["nnUNetv2_train", DATASET_ID, "3d_fullres", "0"]
    
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    # Start training process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Print output in real-time
    print("=" * 80)
    print("Training Output:")
    print("=" * 80)
    
    for line in process.stdout:
        print(line, end='')
        sys.stdout.flush()
    
    process.wait()
    
    if process.returncode == 0:
        print("\n" + "=" * 80)
        print("✅ Training Completed Successfully!")
        print("=" * 80)
        
        # Show final results
        print("\nFinal Checkpoints:")
        if os.path.exists(RESULTS_PATH):
            for f in os.listdir(RESULTS_PATH):
                if f.endswith('.pth'):
                    print(f"  - {f}")
        
        print(f"\nResults saved to: {RESULTS_PATH}")
        print("\nNext steps:")
        print("  1. View training curves: python monitor_training.py final")
        print("  2. Train other folds: nnUNetv2_train 001 3d_fullres 1")
        print("  3. Run inference on test data")
    else:
        print("\n" + "=" * 80)
        print(f"❌ Training Failed with exit code {process.returncode}")
        print("=" * 80)

if __name__ == "__main__":
    try:
        train_nnunet()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        print("You can resume training later with the same command.")
        sys.exit(1)
