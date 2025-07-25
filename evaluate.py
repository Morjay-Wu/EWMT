import numpy as np
import rasterio
from pathlib import Path
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_sam(img_tgt, img_fus):
    """Calculates the Spectral Angle Mapper (SAM).

    Args:
        img_tgt (numpy.ndarray): Target image, typically with shape (C, H, W).
        img_fus (numpy.ndarray): Fused/predicted image, with the same shape as img_tgt.

    Returns:
        float: The mean spectral angle in degrees.
    """
    # Remove any singleton batch dimension
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)

    # Check for correct dimensions (C, H, W)
    if img_tgt.ndim != 3 or img_fus.ndim != 3:
        raise ValueError(f"Input images must have 3 dimensions (C, H, W), but got Tgt: {img_tgt.ndim}, Fus: {img_fus.ndim}")

    C, H, W = img_tgt.shape

    # Reshape images into (H*W, C), where each row is a pixel's spectral vector
    target_pixels = img_tgt.reshape(C, -1).T
    pred_pixels = img_fus.reshape(C, -1).T

    # Calculate dot product for each pixel
    dot_products = np.sum(target_pixels * pred_pixels, axis=1)

    # Calculate L2 norm for each pixel's vector
    norm_target = np.linalg.norm(target_pixels, axis=1)
    norm_pred = np.linalg.norm(pred_pixels, axis=1)

    # Calculate the cosine of the angle, adding epsilon for numerical stability
    epsilon = 1e-8
    denominator = norm_target * norm_pred + epsilon
    cos_angles = dot_products / denominator

    # Clip values to the valid range for arccos, [-1, 1]
    cos_angles = np.clip(cos_angles, -1.0, 1.0)

    # Calculate angles in radians
    angles_rad = np.arccos(cos_angles)

    # Convert to degrees
    angles_deg = angles_rad * 180.0 / np.pi

    # Calculate the mean of all pixel angles
    mean_sam_deg = np.mean(angles_deg)
    return mean_sam_deg

def calculate_ergas(target, pred, ratio=0.3):
    """Calculates the ERGAS (Relative Dimensionless Global Error in Synthesis)."""
    # RMSE per band
    rmse_per_band = np.sqrt(np.mean((target - pred)**2, axis=(1, 2)))
    # Mean value per band
    mean_per_band = np.mean(target, axis=(1, 2))
  
    # Prevent division by zero
    mean_per_band[mean_per_band == 0] = 1e-8
  
    # Calculate the error ratio per band
    error_ratio = (rmse_per_band / mean_per_band)**2
  
    return 100 * ratio * np.sqrt(np.mean(error_ratio))


def calculate_metrics(pred_path, target_path, norm_value=10000):
    """Calculates all evaluation metrics."""
    # Read images
    with rasterio.open(pred_path) as src:
        pred = src.read().astype(np.float32) / norm_value  # (C, H, W)
  
    with rasterio.open(target_path) as src:
        target = src.read().astype(np.float32) / norm_value

    # Verify shapes
    assert pred.shape == target.shape, f"Shape mismatch: Pred {pred.shape} vs Target {target.shape}"
  
    # Clip values to [0, 1] range
    pred = np.clip(pred, 0, 1)
    target = np.clip(target, 0, 1)
  
    # Initialize results dictionary
    metrics = {}
  
    # Calculate RMSE
    metrics['RMSE'] = np.sqrt(np.mean((pred - target)**2))
  
    # Calculate PSNR (averaged over bands)
    psnr_values = [psnr(target[b], pred[b], data_range=1.0) 
                  for b in range(target.shape[0])]
    metrics['PSNR'] = np.mean(psnr_values)
  
    # Calculate SSIM (averaged over bands)
    ssim_values = []
    for b in range(target.shape[0]):
        ssim_val = ssim(target[b], pred[b], data_range=1.0,
                       win_size=11, channel_axis=None)
        ssim_values.append(ssim_val)
    metrics['SSIM'] = np.mean(ssim_values)
  
    # Calculate Correlation Coefficient
    metrics['Correlation'] = np.corrcoef(pred.flatten(), target.flatten())[0,1]
    metrics['CC'] = metrics['Correlation']
  
    # Calculate SAM
    metrics['SAM'] = calculate_sam(target, pred)
  
    # Calculate ERGAS (assuming a resolution ratio, which should be adjusted based on the dataset)
    metrics['ERGAS'] = calculate_ergas(target, pred, ratio=2/35)
  
    return metrics

def batch_evaluate(test_dir, pred_suffix="01_L.tif", gt_suffix="01_L_gt.tif"):
    """Evaluates all samples in a directory in batch."""
    test_dir = Path(test_dir)
    results = []
  
    # Iterate over all subdirectories
    for sample_dir in test_dir.iterdir():
        if not sample_dir.is_dir():
            continue
          
        # Construct file paths
        pred_path = sample_dir / pred_suffix
        gt_path = sample_dir / gt_suffix
      
        # Skip incomplete samples
        if not pred_path.exists() or not gt_path.exists():
            print(f"Skipping incomplete sample: {sample_dir.name}")
            continue
          
        try:
            # Calculate metrics
            metrics = calculate_metrics(str(pred_path), str(gt_path))
            metrics['Sample'] = sample_dir.name
            results.append(metrics)
            print(f"Processed: {sample_dir.name}")
        except Exception as e:
            print(f"Failed to process {sample_dir.name}: {str(e)}")
  
    # Convert to DataFrame
    df = pd.DataFrame(results)
  
    # Calculate and append average values
    avg = df.mean(numeric_only=True) 
    avg['Sample'] = 'Average'
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
  
    return df

if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Users should change this to their local test results directory.
    test_folder = "./results/test"
    output_csv = "./results/test/evaluation_results.csv"

    # Run batch evaluation
    results_df = batch_evaluate(test_folder)
  
    # Print results to console
    print("\n--- Evaluation Summary ---")
    print(f"Test Folder: {test_folder}")
    print(results_df.round(4).to_string(index=False))
  
    # Save results to a CSV file
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults have been saved to: {output_csv}")