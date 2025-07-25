# EWMT: An Enhanced Wavelet and Multi-Head Transformer for Spatiotemporal Fusion

This repository contains the official PyTorch implementation of the paper: **"An Enhanced Wavelet and Multi-Head Transformer for Spatiotemporal Fusion of Remote Sensing Imagery"**.

EWMT is a lightweight, two-input spatiotemporal fusion network that optimizes computational speed and memory footprint while enhancing fusion quality. It leverages a convolution-based wavelet transform and a multi-head attention mechanism to achieve state-of-the-art performance, especially in resource-constrained environments.

![Model Framework](fig/framework.png) 


---

## 🚀 Features

- **High Efficiency**: Designed as a two-input model, EWMT significantly reduces data requirements and computational overhead compared to traditional three- or five-input methods.
- **Advanced Fusion Quality**: Outperforms other two-input models and is competitive with some three-input models, particularly on complex datasets.
- **Lightweight Architecture**: With only ~0.48M parameters, it is suitable for edge computing scenarios like on-board satellite processing.
- **Hybrid Attention Mechanism**: Integrates channel attention and multi-head spatial attention to capture both spectral fidelity and fine-grained spatial details.
- **Optimized Wavelet Transform**: Reformulates the wavelet transform as an efficient convolution operation, enabling seamless GPU acceleration and end-to-end training.

---

## 🔧 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Morjay-Wu/EWMT.git
    cd EWMT
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 📁 Data Preparation

The model expects the dataset to be organized in a specific structure. Based on the CIA dataset example, your data directory should look like this:

```
data/
└── cia/
    ├── train/
    │   ├── L-Mdata1/
    │   │   ├── 00_L.tif  (Fine-resolution reference image at t_k)
    │   │   └── 00_M.tif  (Coarse-resolution reference image at t_k)
    │   │   ├── 01_L.tif  (Fine-resolution target image at t_0 - Ground Truth)
    │   │   └── 01_M.tif  (Coarse-resolution target image at t_0)
    │   ├── L-Mdata2/
    │   │   └── ...
    │   └── ...
    └── test/
        ├── L-Mdata13/
        │   └── ...
        └── ...
```

-   Each subfolder (e.g., `L-Mdata1`) represents one data sample pair.
-   The file names must follow the `time-prefix_sensor-prefix.tif` convention.
    -   `00_`: Reference date (t_k)
    -   `01_`: Prediction date (t_0)
    -   `L`: Fine-resolution sensor (e.g., Landsat)
    -   `M`: Coarse-resolution sensor (e.g., MODIS)

After organizing your data, update the paths in the `run.py` script accordingly.

---

## ⚡️ Quick Start

### Training

To train the EWMT model on your dataset, you can run the `run.py` script. Adjust the arguments as needed.

```bash
python run.py \
    --train_dir ./data/cia/train \
    --val_dir ./data/cia/test \
    --save_dir ./results/cia_experiment \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-3 \
    --cuda
```

-   `--train_dir`: Path to the training dataset.
-   `--val_dir`: Path to the validation dataset.
-   `--save_dir`: Directory to save checkpoints and logs.
-   Other parameters can be found in `run.py`.

Training progress and checkpoints will be saved in the directory specified by `--save_dir`. The best model checkpoint will be saved as `best.pth`.

### Testing

After training, you can use the trained model to generate fused images for the test set.

```bash
python run.py \
    --test_dir ./data/cia/test \
    --save_dir ./results/cia_experiment \
    --epochs 0 \
    --cuda
```
By setting `--epochs 0`, the script will skip the training phase, load the `best.pth` model from the `--save_dir`, and run inference on the test set. The fused images will be saved in the `results/cia_experiment/test` directory.

### Evaluation

To evaluate the performance of the fused images, use the `evaluate.py` script.

```bash
python evaluate.py
```
Make sure to update the `test_folder` path inside the `evaluate.py` script to point to your directory containing the prediction and ground truth images. The script will calculate metrics like RMSE, PSNR, SSIM, SAM, and ERGAS, and save the results to a CSV file.

---

##  citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{wu2025ewmt,
  title={An Enhanced Wavelet and Multi-Head Transformer for Spatiotemporal Fusion of Remote Sensing Imagery},
  author={Wu, Tongquan and Kong, Weiquan and Bai, Lu},
  journal={Journal of Remote Sensing Applications},
  year={2025},
  publisher={MDPI}
}
```

---

## Acknowledgements
The SSIM/MS-SSIM implementation is adapted from [pytorch-msssim](https://github.com/jorge-pessoa/pytorch-msssim). 