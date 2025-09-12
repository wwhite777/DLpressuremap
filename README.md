# DLpressuremap
 Infrared Only Pixel Wise Pressure Mapping For In Bed Monitoring: U Net With Ema Smoothing And Ridge–Morphological Calibration
 
Deep learning model for predicting Pressure Maps (PM) from Infrared (IR) images using U-Net architecture with ResNet50 encoder.

```bash
# Clone repository
git clone <your-repo-url>
cd ICASSP

# Install dependencies
pip install torch torchvision segmentation-models-pytorch
pip install numpy opencv-python scikit-image scikit-learn
pip install pandas matplotlib tqdm pillow
pip install torchmetrics
```
# Project Structure
```bash
~/ICASSP/
├── config.py           # Configuration management
├── utils.py            # Common utilities
├── train_model.py      # Model training script
├── evaluate_metrics.py # Evaluation and baseline methods
├── Depth_IR_PM/        # Data directory
│   ├── IR_png/         # Infrared images
│   └── PM_png/         # Pressure map ground truth
└── result/             # Output directory
    └── model_save/     # Saved models
```
# Data Format
Image files should follow this naming convention:
```bash
{subject}_{modality}_{condition}_image_{frame}.png
```
# Usage
## Training
```bash
# Basic training with default config
python train_model.py

# Custom parameters
python train_model.py --epochs 100 --batch-size 16 --lr 0.001 --run-name experiment1
```
## Evaluation
```bash
# Evaluate with metrics calculation
python evaluate_metrics.py --manifest result/run_default/manifest.json

# Run baseline methods
python evaluate_metrics.py --baseline ridge --cal-frac 0.2
```
# Configuration
Default settings in config.py:

Image size: 192×96 pixels
Batch size: 32
Learning rate: 1e-3
Train/Val/Test split: 80/10/10
Model: U-Net with ResNet50 encoder

Modify config programmatically:
```bash
from config import Config
config = Config()
config.set('training.batch_size', 64)
config.set('training.num_epochs', 100)
```
# Model Architecture
-Encoder: ResNet50 (pretrained on ImageNet)
-Decoder: U-Net decoder
-Input: 3-channel RGB images
-Output: 1-channel pressure maps
-Loss: MSE
-Optimizer: Adam
# Metrics
The model is evaluated using:
-RMSE (Root Mean Square Error)
-SSIM (Structural Similarity Index)
-PSNR (Peak Signal-to-Noise Ratio)
-MAE (Mean Absolute Error)
# Output
Training produces:
-Model checkpoint: result/model_save/IR_PM_save/model_e{epoch}.pth
-Predictions: result/{run_name}/preds/*.npy
-Metrics report: Console output with test set performance
# Requirements
-Python 3.7+
-PyTorch 1.9+
-CUDA-capable GPU (recommended)
-~4GB RAM minimum
