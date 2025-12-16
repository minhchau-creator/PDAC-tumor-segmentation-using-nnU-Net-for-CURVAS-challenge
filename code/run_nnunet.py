"""
Run nnU-Net pipeline: verify dataset, preprocess, plan, and train
This script handles all nnU-Net steps with proper environment setup
Includes real-time monitoring of training progress
"""

import os
import subprocess
import sys
import time
import threading

# Setup nnU-Net environment variables
BASE_PATH = "/home/minhchau/anaconda3/envs/curvas/curvas"
os.environ['nnUNet_raw'] = f"{BASE_PATH}/nnUNet_raw"
os.environ['nnUNet_preprocessed'] = f"{BASE_PATH}/nnUNet_preprocessed"
os.environ['nnUNet_results'] = f"{BASE_PATH}/nnUNet_results"

# Create directories if not exist
os.makedirs(os.environ['nnUNet_preprocessed'], exist_ok=True)
os.makedirs(os.environ['nnUNet_results'], exist_ok=True)

print("=" * 80)
print("nnU-Net Environment Setup")
print("=" * 80)
print(f"nnUNet_raw: {os.environ['nnUNet_raw']}")
print(f"nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
print(f"nnUNet_results: {os.environ['nnUNet_results']}")
print("")

DATASET_ID = "001"
RESULTS_PATH = f"{BASE_PATH}/nnUNet_results/Dataset{DATASET_ID}_PDAC/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0"

def read_training_metrics():
    """Read latest training metrics from log file"""
    log_file = os.path.join(RESULTS_PATH, "training_log.txt")
    
    if not os.path.exists(log_file):
        return None
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Parse metrics from recent lines
        metrics = {'epoch': '?', 'loss': '?', 'dice': '?'}
        for line in reversed(lines[-30:]):
            line_lower = line.lower()
            
            # Look for epoch
            if 'epoch' in line_lower and metrics['epoch'] == '?':
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'epoch' in part.lower() and i+1 < len(parts):
                        metrics['epoch'] = parts[i+1].rstrip(':,')
                        break
            
            # Look for loss
            if 'loss' in line_lower and metrics['loss'] == '?':
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'loss' in part.lower() and i+1 < len(parts):
                        try:
                            metrics['loss'] = f"{float(parts[i+1]):.4f}"
                        except:
                            pass
                        break
            
            # Look for dice/validation metrics
            if ('dice' in line_lower or 'validation' in line_lower) and metrics['dice'] == '?':
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'dice' in part.lower() and i+1 < len(parts):
                        try:
                            metrics['dice'] = f"{float(parts[i+1]):.4f}"
                        except:
                            pass
                        break
            
            # Stop if we have enough info
            if all(v != '?' for v in metrics.values()):
                break
        
        return metrics
    except:
        return None

def monitor_training_progress(stop_event):
    """Monitor training in background thread"""
    print("\n" + "=" * 80)
    print("📊 Training Monitor Active (updating every 10 seconds)")
    print("=" * 80)
    
    last_update = ""
    while not stop_event.is_set():
        time.sleep(10)
        
        metrics = read_training_metrics()
        if metrics:
            update = f"[Epoch: {metrics['epoch']} | Loss: {metrics['loss']} | Dice: {metrics['dice']}]"
            if update != last_update:
                print(f"\r{update}", end='', flush=True)
                last_update = update
        
        # Check for checkpoints
        if os.path.exists(RESULTS_PATH):
            checkpoints = [f for f in os.listdir(RESULTS_PATH) if f.endswith('.pth')]
            if checkpoints and 'checkpoint_final.pth' in checkpoints:
                print("\n\n🎉 Training Complete!")
                break
    
    print("\n" + "=" * 80)

def run_command(cmd, description, monitor=False):
    """Run a command and handle errors, with optional monitoring"""
    print("=" * 80)
    print(f"Step: {description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print("")
    
    try:
        # Start monitoring thread if requested
        stop_event = threading.Event()
        monitor_thread = None
        
        if monitor:
            monitor_thread = threading.Thread(target=monitor_training_progress, args=(stop_event,))
            monitor_thread.daemon = True
            monitor_thread.start()
        
        # Run the command
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=False  # Show output in real-time
        )
        
        # Stop monitoring
        if monitor_thread:
            stop_event.set()
            monitor_thread.join(timeout=2)
        
        print(f"\n✅ {description} - COMPLETED\n")
        return True
    except subprocess.CalledProcessError as e:
        if monitor_thread:
            stop_event.set()
        print(f"\n❌ {description} - FAILED")
        print(f"Error code: {e.returncode}")
        return False
    except Exception as e:
        if monitor_thread:
            stop_event.set()
        print(f"\n❌ {description} - ERROR: {str(e)}")
        return False

def main():
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        steps = sys.argv[1].split(',')  # e.g., "verify,preprocess,train" or "all"
    else:
        # Default: run all steps
        steps = ['all']
    
    run_verify = 'verify' in steps or 'all' in steps
    run_preprocess = 'preprocess' in steps or 'all' in steps
    run_train = 'train' in steps or 'all' in steps
    
    # Step 1: Verify dataset integrity
    if run_verify:
        print("\n" + "=" * 80)
        print("STEP 1: Verify Dataset Integrity")
        print("=" * 80)
        success = run_command(
            ["nnUNetv2_plan_and_preprocess", "-d", DATASET_ID, "--verify_dataset_integrity"],
            "Dataset Integrity Check"
        )
        if not success:
            print("⚠️  Dataset verification failed but continuing...")
    
    # Step 2: Run nnU-Net preprocessing and planning
    if run_preprocess:
        print("\n" + "=" * 80)
        print("STEP 2: nnU-Net Preprocessing and Planning")
        print("=" * 80)
        print("This will:")
        print("  - Analyze intensity distribution and spacing")
        print("  - Determine optimal resampling strategy")
        print("  - Calculate patch size based on GPU memory")
        print("  - Create preprocessing plans")
        print("  - Preprocess all training cases")
        print("")
        success = run_command(
            ["nnUNetv2_plan_and_preprocess", "-d", DATASET_ID, "-c", "3d_fullres"],
            "nnU-Net Preprocessing and Planning"
        )
        if not success:
            print("❌ Preprocessing failed. Cannot continue to training.")
            return
    
    # Step 3: Train nnU-Net
    if run_train:
        print("\n" + "=" * 80)
        print("STEP 3: Train nnU-Net")
        print("=" * 80)
        print("Training configuration:")
        print("  - Dataset: 001 (PDAC)")
        print("  - Configuration: 3d_fullres")
        print("  - Fold: 0 (first fold of 5-fold cross-validation)")
        print("  - GPU: Will use CUDA automatically")
        print("")
        print("⚠️  Training can take several hours or days depending on:")
        print("  - Number of epochs (default: 1000)")
        print("  - Dataset size (40 cases)")
        print("  - GPU speed (RTX 4090)")
        print("")
        print("💡 Tip: Training monitor will show progress every 10 seconds")
        print("")

        # Check for checkpoint files
        checkpoint_latest = os.path.join(RESULTS_PATH, "checkpoint_latest.pth")
        checkpoint_best = os.path.join(RESULTS_PATH, "checkpoint_best.pth")
        continue_training = False
        if os.path.exists(checkpoint_latest) or os.path.exists(checkpoint_best):
            continue_training = True
            print("🔄 Found checkpoint file(s), will resume training from last epoch.")
            print(f"   Latest checkpoint: {checkpoint_latest}")

        train_cmd = ["nnUNetv2_train", DATASET_ID, "3d_fullres", "0"]
        if continue_training:
            train_cmd.append("--c")
            print("   Adding --c flag to continue training from checkpoint.")

        success = run_command(
            train_cmd,
            "nnU-Net Training (Fold 0)",
            monitor=True  # Enable monitoring for training
        )
        if success:
            print("\n" + "=" * 80)
            print("🎉 TRAINING COMPLETED!")
            print("=" * 80)
            print(f"Model saved to: {RESULTS_PATH}/")

            # Show final metrics
            final_metrics = read_training_metrics()
            if final_metrics:
                print("\n📊 Final Metrics:")
                print(f"   Epoch: {final_metrics['epoch']}")
                print(f"   Loss: {final_metrics['loss']}")
                print(f"   Dice: {final_metrics['dice']}")

            # List checkpoints
            if os.path.exists(RESULTS_PATH):
                checkpoints = [f for f in os.listdir(RESULTS_PATH) if f.endswith('.pth')]
                if checkpoints:
                    print(f"\n💾 Checkpoints saved ({len(checkpoints)}):")
                    for ckpt in sorted(checkpoints)[-3:]:  # Show last 3
                        print(f"   - {ckpt}")

            # Check for progress plot
            progress_plot = os.path.join(RESULTS_PATH, "progress.png")
            if os.path.exists(progress_plot):
                print(f"\n📈 Training curves: {progress_plot}")

            print("")
            print("Next steps:")
            print("  1. Train other folds: python run_nnunet.py train (change fold in script)")
            print("  2. Run inference: nnUNetv2_predict -i <input_folder> -o <output_folder> -d 001 -c 3d_fullres -f 0")
            print("  3. Evaluate results: nnUNetv2_evaluate_folder <gt_folder> <pred_folder>")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user (Ctrl+C)")
        sys.exit(1)
