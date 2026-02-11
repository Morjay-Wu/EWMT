# EWMT: Wavelet-based Spatiotemporal Fusion

This repository contains the implementation of **EWMT**, a deep learning model for spatiotemporal fusion of remote sensing images. The model integrates SwinIR-style residual blocks with wavelet transforms to effectively fuse high-temporal-resolution (e.g., MODIS) and high-spatial-resolution (e.g., Landsat) images.

## Project Structure

- **`model_EWMT.py`**: Core model definition (`FusionNet_2`), including `RFEncoder` (SwinResidualRFBlock) and Wavelet Transform modules.
- **`train.py`**: Training script tailored for the CIA dataset.
- **`evaluate.py`**: Evaluation script providing metrics such as RMSE, PSNR, SSIM, SAM, and ERGAS.
- **`wt_conv.py` & `wavelet.py`**: Utilities for Wavelet Transform operations (Haar Wavelet).
- **`ssim.py`**: Implementation of SSIM/MS-SSIM for loss calculation.
- **`utils.py`**: General helper functions.

## Requirements

The code requires a Python environment with PyTorch and common remote sensing/image processing libraries:

- Python 3.x
- PyTorch (CUDA recommended)
- Rasterio
- NumPy
- Pandas
- Scikit-image
- PyWavelets (`pywt`)
- Tqdm

## Usage

### Training

Use `train.py` to train the model.

```bash
python train.py --data_dir /path/to/dataset --out_dir /path/to/output
```

**Common Arguments:**
- `--data_dir`: Root directory of the dataset.
- `--out_dir`: Directory to save checkpoints and logs.
- `--epochs`: Number of training epochs (default: 100).
- `--batch_size`: Batch size (default: 8).
- `--lr`: Learning rate (default: 1e-4).

### Evaluation

Use `evaluate.py` to calculate quantitative metrics on test results.

**Note:** Please check and modify the `test_folder` location in `evaluate.py` before running.

```bash
python evaluate.py
```

## Note on Missing Dependencies

The current file list does not include `data_cia.py` or `dwt_def.py`, which are imported by `train.py` and `model_EWMT.py`. Please ensure these files are present in your environment or directory for the code to execute correctly.
