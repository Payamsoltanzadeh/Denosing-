# ============================================================================
# test_denoising.py
# ============================================================================
# Date: 2025-12-05
# Purpose: Inference and Visualization for Simple U-Net Denoiser.
#          Loads trained model, predicts dose, and visualizes results.
#          Calculates RMSE and MAE.
# ============================================================================

import os
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Set backend to non-interactive
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import Model and Dataset
from models.simple_unet_denoiser import get_simple_denoiser
from dataset.pl_dose_dataset import ConditionalDoseDataset
from utils.gamma_index import calculate_gamma_index_3d

def compute_metrics(pred, target):
    """Compute RMSE, MAE, and Gamma Pass Rates (3%/3mm, 2%/2mm, 1%/1mm)."""
    mse = np.mean((pred - target) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - target))
    
    # Gamma 3%/3mm
    _, gamma_33 = calculate_gamma_index_3d(target, pred, dta_mm=3.0, dd_percent=3.0)
    
    # Gamma 2%/2mm
    _, gamma_22 = calculate_gamma_index_3d(target, pred, dta_mm=2.0, dd_percent=2.0)
    
    # Gamma 1%/1mm
    _, gamma_11 = calculate_gamma_index_3d(target, pred, dta_mm=1.0, dd_percent=1.0)
    
    return rmse, mae, gamma_33, gamma_22, gamma_11

def visualize_results(ct, lp, hp, pred, save_path, sample_idx):
    """
    Visualize central slice of CT, LP, HP, and Prediction.
    Saves the figure to save_path.
    """
    # Take central slice
    d_center = ct.shape[0] // 2
    
    ct_slice = ct[d_center, :, :]
    lp_slice = lp[d_center, :, :]
    hp_slice = hp[d_center, :, :]
    pred_slice = pred[d_center, :, :]
    diff_slice = np.abs(hp_slice - pred_slice)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # CT
    im0 = axes[0].imshow(ct_slice, cmap='gray')
    axes[0].set_title("CT (Anatomy)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # LP (Input)
    im1 = axes[1].imshow(lp_slice, cmap='jet')
    axes[1].set_title("LP Dose (Noisy Input)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # HP (Target)
    im2 = axes[2].imshow(hp_slice, cmap='jet')
    axes[2].set_title("HP Dose (Target)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Prediction
    im3 = axes[3].imshow(pred_slice, cmap='jet')
    axes[3].set_title("Predicted Dose")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    # Difference
    im4 = axes[4].imshow(diff_slice, cmap='hot')
    axes[4].set_title("Abs Difference")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"sample_{sample_idx}.png"), dpi=150)
    plt.close()

def test(args):
    # ------------------------------------------------------------------------
    # 1. Setup
    # ------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "gpu" else "cpu")
    print(f"🚀 Testing on: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ------------------------------------------------------------------------
    # 2. Load Model
    # ------------------------------------------------------------------------
    print(f"🏗️ Loading model from: {args.model_path}")
    model = get_simple_denoiser(
        model_type=args.model_type,
        base_channels=args.base_channels
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # ------------------------------------------------------------------------
    # 3. Load Data
    # ------------------------------------------------------------------------
    print(f"📂 Loading dataset from: {args.root_dir}")
    # Use same parameters as training
    dataset = ConditionalDoseDataset(
        root_dir=args.root_dir,
        normalize=True,
        target_dim=64,
        num_photons=1e3,
        add_noise=False, # Use pre-generated LP only
        use_pregenerated_lp=True,
        lp_folder=args.lp_folder
    )
    
    # Use a subset for testing if requested
    if args.num_samples > 0:
        indices = list(range(min(len(dataset), args.num_samples)))
        dataset = torch.utils.data.Subset(dataset, indices)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(f"   Test samples: {len(dataset)}")
    
    # ------------------------------------------------------------------------
    # 4. Inference Loop
    # ------------------------------------------------------------------------
    print("🔥 Starting inference...")
    
    all_rmse = []
    all_mae = []
    all_gamma_33 = []
    all_gamma_22 = []
    all_gamma_11 = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            hp_target, condition = batch
            
            ct = condition['ct'].to(device)
            lp = condition['lp_dose'].to(device)
            hp_target = hp_target.to(device)
            
            # Forward
            inputs = torch.cat([ct, lp], dim=1)
            hp_pred = model(inputs)
            
            # Convert to numpy for metrics and viz
            # Shape: [D, H, W]
            ct_np = ct[0, 0].cpu().numpy()
            lp_np = lp[0, 0].cpu().numpy()
            hp_np = hp_target[0, 0].cpu().numpy()
            pred_np = hp_pred[0, 0].cpu().numpy()
            
            # Compute Metrics
            rmse, mae, g33, g22, g11 = compute_metrics(pred_np, hp_np)
            all_rmse.append(rmse)
            all_mae.append(mae)
            all_gamma_33.append(g33)
            all_gamma_22.append(g22)
            all_gamma_11.append(g11)
            
            print(f"   Sample {i}: RMSE={rmse:.4f}, MAE={mae:.4f} | Gamma: 3%/3mm={g33:.1f}%, 2%/2mm={g22:.1f}%, 1%/1mm={g11:.1f}%")
            
            # Visualize
            visualize_results(ct_np, lp_np, hp_np, pred_np, args.output_dir, i)
            
            # Save raw volumes (optional)
            if args.save_volumes:
                np.save(os.path.join(args.output_dir, f"sample_{i}_pred.npy"), pred_np)
                np.save(os.path.join(args.output_dir, f"sample_{i}_target.npy"), hp_np)

    # ------------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------------
    avg_rmse = np.mean(all_rmse)
    avg_mae = np.mean(all_mae)
    avg_g33 = np.mean(all_gamma_33)
    avg_g22 = np.mean(all_gamma_22)
    avg_g11 = np.mean(all_gamma_11)
    
    print("\n" + "="*60)
    print(f"📊 Final Results ({len(dataset)} samples)")
    print(f"   Average RMSE:      {avg_rmse:.6f}")
    print(f"   Average MAE:       {avg_mae:.6f}")
    print(f"   Avg Gamma (3%/3mm): {avg_g33:.2f}%")
    print(f"   Avg Gamma (2%/2mm): {avg_g22:.2f}%")
    print(f"   Avg Gamma (1%/1mm): {avg_g11:.2f}%")
    print(f"   Visualizations saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="Mini_Dataset")
    parser.add_argument("--model_path", type=str, default="results/simple_unet/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="results/predictions")
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--model_type", type=str, default="standard")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to test (0 for all)")
    parser.add_argument("--save_volumes", action="store_true", help="Save 3D numpy volumes")
    parser.add_argument("--lp_folder", type=str, default="lp_cubes", help="Folder name for LP dose")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found at {args.model_path}")
        print("   Run simple_train_denoising.py first!")
    else:
        test(args)
