# CURVAS PDAC Segmentation

Pancreatic Ductal Adenocarcinoma (PDAC) tumor segmentation using nnU-Net for the CURVAS-PDACVI Challenge.

## Project Structure

```
curvas/
├── code/
│   ├── almost_no_preprocess.py    # Data preprocessing and nnU-Net format conversion
│   ├── run_nnunet.py              # Training script with monitoring
│   └── test_model.py              # Inference and evaluation script
├── Data/                          # Original CURVAS dataset (not included)
├── nnUNet_raw/                    # nnU-Net formatted data (generated)
├── nnUNet_preprocessed/           # Preprocessed data (generated)
└── nnUNet_results/                # Training results and checkpoints (generated)
```

## Requirements

- Python 3.10+
- PyTorch with CUDA support
- nnU-Net v2
- MONAI
- SimpleITK
- NumPy

## Installation

```bash
conda create -n curvas python=3.10
conda activate curvas
pip install nnunetv2 monai SimpleITK numpy matplotlib
```

## Usage

### 1. Data Preprocessing

```bash
python code/almost_no_preprocess.py
```

### 2. Training

```bash
python code/run_nnunet.py train
```

### 3. Inference and Evaluation

```bash
python code/test_model.py
```

## Challenge Metrics

This project evaluates models using CURVAS-PDACVI challenge metrics:

1. **Quality of Segmentation**
   - Classic Dice Score (DSC)
   - Thresholding Dice Score (thresh-DSC)

2. **Multi-Rater Calibration**
   - Expected Calibration Error (ECE)

3. **Volume Assessment**
   - Continuous Ranked Probability Score (CRPS)

4. **Vascular Invasion**
   - Wasserstein Distance for 5 vascular structures

## Results

Training progress and final Dice score: ~0.75-0.76 on validation set.

## License

MIT License

## Acknowledgments

- CURVAS-PDACVI Challenge organizers
- nnU-Net framework by Fabian Isensee et al.
