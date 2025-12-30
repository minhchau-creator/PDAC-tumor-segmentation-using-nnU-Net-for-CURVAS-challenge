"""
Comprehensive evaluation of nnU-Net predictions for PDAC detection
Calculates detailed metrics and generates visualizations

Usage:
    python evaluate_predictions.py --input /path/to/predictions --output /path/to/output
    python evaluate_predictions.py  (will use default paths)
"""

import os
import sys
import json
import argparse
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import label as scipy_label
from scipy.spatial.distance import directed_hausdorff

def calculate_dice(pred, gt):
    """Calculate Dice coefficient"""
    intersection = np.sum(pred * gt)
    if pred.sum() + gt.sum() == 0:
        return 1.0  # Both empty
    return 2.0 * intersection / (pred.sum() + gt.sum())

def calculate_iou(pred, gt):
    """Calculate Intersection over Union"""
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    if union == 0:
        return 1.0
    return intersection / union

def calculate_sensitivity_specificity(pred, gt):
    """Calculate sensitivity and specificity"""
    tp = np.sum((pred == 1) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return sensitivity, specificity

def calculate_hausdorff(pred, gt):
    """Calculate Hausdorff distance"""
    pred_points = np.argwhere(pred > 0)
    gt_points = np.argwhere(gt > 0)
    
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float('inf')
    
    hd1 = directed_hausdorff(pred_points, gt_points)[0]
    hd2 = directed_hausdorff(gt_points, pred_points)[0]
    
    return max(hd1, hd2)

def evaluate_single_case(pred_path, gt_path, case_name):
    """Evaluate a single case"""
    # Load images using SimpleITK
    pred_img = sitk.ReadImage(pred_path)
    gt_img = sitk.ReadImage(gt_path)

    pred = sitk.GetArrayFromImage(pred_img)
    gt = sitk.GetArrayFromImage(gt_img)

    # Binarize (in case there are multiple labels)
    pred_binary = (pred > 0).astype(np.uint8)
    gt_binary = (gt > 0).astype(np.uint8)

    # Calculate metrics
    dice = calculate_dice(pred_binary, gt_binary)
    iou = calculate_iou(pred_binary, gt_binary)
    sensitivity, specificity = calculate_sensitivity_specificity(pred_binary, gt_binary)

    # Volume metrics (using spacing)
    spacing = pred_img.GetSpacing()
    voxel_volume = spacing[0] * spacing[1] * spacing[2]

    gt_volume = np.sum(gt_binary) * voxel_volume
    pred_volume = np.sum(pred_binary) * voxel_volume
    volume_error = abs(pred_volume - gt_volume) / gt_volume if gt_volume > 0 else 0
    
    # Hausdorff distance
    try:
        hd = calculate_hausdorff(pred_binary, gt_binary)
    except:
        hd = float('inf')
    
    # Detection status
    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    has_tumor = np.sum(gt_binary) > 0
    detected = tp > 0
    
    return {
        'case': case_name,
        'dice': dice,
        'iou': iou,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'hausdorff': hd,
        'gt_volume_mm3': float(gt_volume),
        'pred_volume_mm3': float(pred_volume),
        'volume_error': volume_error,
        'has_tumor': has_tumor,
        'detected': detected
    }

def generate_report(results, output_dir):
    """Generate evaluation report"""
    
    # Calculate overall statistics
    metrics = ['dice', 'iou', 'sensitivity', 'specificity', 'volume_error']
    stats = {}
    
    for metric in metrics:
        values = [r[metric] for r in results if not np.isinf(r.get('hausdorff', 0))]
        stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
    
    # Detection rate
    total_with_tumor = sum(1 for r in results if r['has_tumor'])
    total_detected = sum(1 for r in results if r['detected'])
    detection_rate = total_detected / total_with_tumor if total_with_tumor > 0 else 0
    
    # Print report
    print("\n" + "="*80)
    print("PDAC DETECTION EVALUATION REPORT")
    print("="*80)
    
    print(f"\n📊 Overall Statistics (n={len(results)} cases):")
    print("-" * 80)
    
    for metric in metrics:
        print(f"\n{metric.upper()}:")
        print(f"  Mean ± Std: {stats[metric]['mean']:.4f} ± {stats[metric]['std']:.4f}")
        print(f"  Median:     {stats[metric]['median']:.4f}")
        print(f"  Range:      [{stats[metric]['min']:.4f}, {stats[metric]['max']:.4f}]")
    
    print(f"\n🎯 Detection Performance:")
    print(f"  Cases with tumor:  {total_with_tumor}")
    print(f"  Successfully detected: {total_detected}")
    print(f"  Detection rate:    {detection_rate:.2%}")
    
    # Per-case results
    print("\n" + "="*80)
    print("Per-Case Results")
    print("="*80)
    print(f"{'Case':<20} {'Dice':<8} {'IoU':<8} {'Sens':<8} {'Spec':<8} {'Vol Err':<10} {'Detected':<10}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: x['dice'], reverse=True):
        detected_str = "✅ Yes" if r['detected'] else "❌ No"
        print(f"{r['case']:<20} {r['dice']:<8.4f} {r['iou']:<8.4f} {r['sensitivity']:<8.4f} "
              f"{r['specificity']:<8.4f} {r['volume_error']:<10.4f} {detected_str:<10}")
    
    # Save results to JSON (convert numpy types to Python types)
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    
    # Convert numpy types to native Python types
    def convert_to_python_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    with open(results_file, 'w') as f:
        json.dump(convert_to_python_types({
            'overall_statistics': stats,
            'detection_rate': detection_rate,
            'per_case_results': results
        }), f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    return stats, results

def plot_results(results, output_dir):
    """Generate visualization plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('PDAC Detection Model Evaluation', fontsize=16, fontweight='bold')
    
    # Extract metrics
    cases = [r['case'] for r in results]
    dice_scores = [r['dice'] for r in results]
    iou_scores = [r['iou'] for r in results]
    sensitivity = [r['sensitivity'] for r in results]
    specificity = [r['specificity'] for r in results]
    volume_errors = [r['volume_error'] for r in results]
    
    # Plot 1: Dice scores
    ax = axes[0, 0]
    ax.barh(range(len(cases)), dice_scores, color='skyblue')
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([c.replace('CURVASPDAC_', '').replace('PDAC_', '') for c in cases], fontsize=8)
    ax.set_xlabel('Dice Score')
    ax.set_title('Dice Coefficient per Case')
    ax.axvline(x=np.mean(dice_scores), color='red', linestyle='--', label=f'Mean: {np.mean(dice_scores):.3f}')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    # Plot 2: IoU scores
    ax = axes[0, 1]
    ax.barh(range(len(cases)), iou_scores, color='lightgreen')
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([c.replace('CURVASPDAC_', '').replace('PDAC_', '') for c in cases], fontsize=8)
    ax.set_xlabel('IoU Score')
    ax.set_title('Intersection over Union per Case')
    ax.axvline(x=np.mean(iou_scores), color='red', linestyle='--', label=f'Mean: {np.mean(iou_scores):.3f}')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 3: Sensitivity vs Specificity
    ax = axes[0, 2]
    ax.scatter(sensitivity, specificity, s=100, alpha=0.6, c=dice_scores, cmap='viridis')
    ax.set_xlabel('Sensitivity')
    ax.set_ylabel('Specificity')
    ax.set_title('Sensitivity vs Specificity')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Dice Score')
    
    # Plot 4: Volume error
    ax = axes[1, 0]
    colors = ['green' if e < 0.2 else 'orange' if e < 0.5 else 'red' for e in volume_errors]
    ax.barh(range(len(cases)), volume_errors, color=colors)
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([c.replace('CURVASPDAC_', '').replace('PDAC_', '') for c in cases], fontsize=8)
    ax.set_xlabel('Relative Volume Error')
    ax.set_title('Volume Prediction Error')
    ax.axvline(x=np.mean(volume_errors), color='black', linestyle='--', label=f'Mean: {np.mean(volume_errors):.3f}')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 5: Metric distribution
    ax = axes[1, 1]
    metrics_data = [dice_scores, iou_scores, sensitivity, specificity]
    ax.boxplot(metrics_data, labels=['Dice', 'IoU', 'Sens', 'Spec'])
    ax.set_ylabel('Score')
    ax.set_title('Metric Distributions')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 6: Detection summary
    ax = axes[1, 2]
    detected_count = sum(1 for r in results if r['detected'])
    missed_count = len(results) - detected_count
    ax.pie([detected_count, missed_count], 
           labels=[f'Detected\n({detected_count})', f'Missed\n({missed_count})'],
           autopct='%1.1f%%',
           colors=['lightgreen', 'lightcoral'],
           startangle=90)
    ax.set_title('PDAC Detection Rate')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'evaluation_plots.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plots saved to: {plot_file}")
    
    plt.close()

def parse_args():
    """Parse command line arguments"""
    BASE_PATH = "/home/minhchau/anaconda3/envs/curvas/curvas"
    DEFAULT_INPUT = f"{BASE_PATH}/test_results/task7_results"
    DEFAULT_OUTPUT = f"{BASE_PATH}/evaluation_results"

    parser = argparse.ArgumentParser(
        description='Evaluate PDAC detection predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate test_model_task7.py results (default)
  python evaluate_predictions.py

  # Custom input/output paths
  python evaluate_predictions.py --input /path/to/predictions --output /path/to/output
        """
    )
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT,
                        help=f'Input directory containing prediction folders (default: {DEFAULT_INPUT})')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help=f'Output directory for evaluation results (default: {DEFAULT_OUTPUT})')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    PREDICTIONS_DIR = args.input
    OUTPUT_DIR = args.output

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*80)
    print("Evaluating PDAC Detection Model")
    print("="*80)
    print(f"Input: {PREDICTIONS_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print("")

    # Check if input exists
    if not os.path.exists(PREDICTIONS_DIR):
        print(f"❌ Error: Input directory does not exist: {PREDICTIONS_DIR}")
        sys.exit(1)

    # Get case folders (e.g., PDAC_0001, PDAC_0002, ...)
    case_folders = sorted([d for d in os.listdir(PREDICTIONS_DIR)
                          if os.path.isdir(os.path.join(PREDICTIONS_DIR, d))])

    if not case_folders:
        print(f"❌ No case folders found in {PREDICTIONS_DIR}")
        print("Expected structure: {input_dir}/PDAC_XXXX/")
        sys.exit(1)

    print(f"Found {len(case_folders)} cases to evaluate\n")

    # Evaluate each case
    results = []
    for case_name in case_folders:
        case_dir = os.path.join(PREDICTIONS_DIR, case_name)

        # Expected file names
        pred_file = os.path.join(case_dir, f"{case_name}_predict.nii.gz")
        gt_file = os.path.join(case_dir, f"{case_name}_groundtruth.nii.gz")

        if not os.path.exists(pred_file):
            print(f"⚠️  Missing prediction for {case_name}, skipping...")
            continue

        if not os.path.exists(gt_file):
            print(f"⚠️  Missing ground truth for {case_name}, skipping...")
            continue

        print(f"Evaluating {case_name}...")
        result = evaluate_single_case(pred_file, gt_file, case_name)
        results.append(result)

    if not results:
        print("❌ No cases were evaluated!")
        print("\nExpected file structure:")
        print("  {input_dir}/PDAC_XXXX/")
        print("    ├── PDAC_XXXX_predict.nii.gz")
        print("    └── PDAC_XXXX_groundtruth.nii.gz")
        sys.exit(1)

    # Generate report
    stats, results = generate_report(results, OUTPUT_DIR)

    # Generate plots
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)
    plot_results(results, OUTPUT_DIR)

    print("\n" + "="*80)
    print("✅ Evaluation Complete!")
    print("="*80)
    print(f"\n📁 Results saved to: {OUTPUT_DIR}/")
    print(f"   - evaluation_results.json (detailed metrics)")
    print(f"   - evaluation_plots.png (visualizations)")

if __name__ == "__main__":
    main()
