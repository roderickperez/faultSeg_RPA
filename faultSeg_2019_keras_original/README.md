# FaultSeg - Original Keras Implementation (Paper-Aligned)

This directory contains the original Keras implementation of the 3D seismic fault segmentation model, as described in the paper "FaultSeg3D: Using synthetic data sets to train an end-to-end convolutional neural network for 3D seismic fault segmentation" by Wu et al. (2019).

The code in this directory has been verified and corrected to ensure it aligns with the parameters specified in the paper.

## Model & Training Parameters

The following parameters are based on the paper and have been implemented in the code:

| Parameter | Value | File |
| :--- | :--- | :--- |
| **Model Architecture** | | `unet3.py` |
| U-Net Type | Simplified 3D U-Net | `unet3.py` |
| Total Conv. Layers | 15 | `unet3.py` |
| Convolution | 3x3x3 with ReLU | `unet3.py` |
| Downsampling | 2x2x2 Max Pooling | `unet3.py` |
| Upsampling | 2x2x2 UpSampling3D | `unet3.py` |
| Final Activation | Sigmoid | `unet3.py` |
| **Training** | | `train.py` |
| Optimizer | Adam | `train.py` |
| Learning Rate | 0.0001 | `train.py` |
| Loss Function | Balanced Binary Cross-Entropy | `train.py` |
| Epochs | 25 | `train.py` |
| Batch Size | 4 (per generator call) | `train.py`, `utils.py` |
| **Data** | | `train.py`, `utils.py` |
| Training Set Size | 200 synthetic pairs | `train.py` |
| Validation Set Size | 20 synthetic pairs | `train.py` |
| Input Image Size | 128x128x128 | `train.py` |
| Normalization | Per-image (Mean/Std Dev) | `utils.py` |
| Data Augmentation | Rotations (0, 90, 180, 270 deg) | `utils.py` |