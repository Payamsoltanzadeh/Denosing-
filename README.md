# 3D U-Net Denoising for Monte Carlo Dose Distributions

This project implements a deep learning pipeline to denoise **Low-Photon (LP) Monte Carlo dose distributions** using a **3D U-Net**. The goal is to accelerate Monte Carlo simulations by computing a noisy, low-photon dose and restoring it to high-photon (HP) quality using the neural network, conditioned on the patient's CT anatomy.

## 🎯 Project Objective
- **Input**: 
  1. **CT Volume** (Anatomy context)
  2. **Low-Photon Dose** (Noisy input, simulated via Poisson statistics)
- **Output**: 
  - **High-Photon Dose** (Clean ground truth)
- **Method**: 
  - Standard 3D U-Net (Encoder-Decoder with Skip Connections).
  - Trained with MSE Loss.
  - Evaluation using RMSE, MAE, and **Gamma Index (3%)**.

## 📂 Project Structure

```
├── dataset/
│   └── pl_dose_dataset.py       # Custom PyTorch Dataset (Loads CT, LP, HP)
├── models/
│   └── simple_unet_denoiser.py  # 3D U-Net Architecture
├── utils/
│   └── gamma_index.py           # Gamma Index Calculation (Medical Physics Metric)
├── generate_lp_dose.py          # Pre-processing: Adds Poisson noise to HP dose
├── simple_train_denoising.py    # Main Training Script (Pure PyTorch)
├── test_denoising.py            # Inference, Visualization & Evaluation Script
└── requirements.txt             # Dependencies
```

## 🚀 Usage Instructions

### 1. Environment Setup
Ensure you have Python installed with the required libraries:
```bash
pip install torch numpy matplotlib scipy tqdm
```

### 2. Data Generation (Simulate Low-Photon Dose)
Before training, generate the noisy "Low-Photon" input data from the clean "High-Photon" ground truth using Poisson statistics.
```bash
# N=1e3 photons (High noise, ~1000x speedup simulation)
python generate_lp_dose.py --num_photons 1e3
```

### 3. Training
Train the U-Net model. The script saves the best model to `results/simple_unet/best_model.pth`.
```bash
# Train for 50 epochs on GPU
python simple_train_denoising.py --epochs 50 --device gpu --batch_size 2
```

### 4. Testing & Evaluation
Run inference on the test set. This script calculates **RMSE**, **MAE**, and **Gamma Index (3%)**, and saves visualization plots.
```bash
python test_denoising.py --model_path results/simple_unet/best_model.pth
```
*Output images will be saved in `results/predictions/`.*

## 📊 Evaluation Metrics
- **RMSE (Root Mean Squared Error)**: Measures global pixel-wise error.
- **MAE (Mean Absolute Error)**: Measures average absolute deviation.
- **Gamma Index (3%)**: Clinical metric for dose distribution comparison. Checks if the predicted dose is within 3% of the ground truth (ignoring spatial shifts for this denoising task).

## 🛠 Technical Details
- **Framework**: PyTorch (Pure implementation, no Lightning dependencies for compatibility).
- **Architecture**: 3D U-Net with 2 input channels (CT, LP) and 1 output channel (HP).
- **Normalization**: Z-score normalization for CT and Dose volumes.
- **Noise Model**: Poisson distribution ($N \approx 10^3$) applied to High-Photon dose.

---
*Developed for Fast Monte Carlo Dose Calculation Research.*
