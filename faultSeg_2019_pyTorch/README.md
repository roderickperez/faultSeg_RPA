# FaultSeg: PyTorch Implementation

This directory contains a PyTorch implementation of the U-Net model for seismic fault segmentation, based on the original work by Wu et al. (2019). The code has been written to be compatible with modern PyTorch versions and follows common PyTorch conventions.

## Code Description

The PyTorch implementation is organized into the following files:

- **`train_pytorch.py`**: This script is used to train the U-Net model. It sets up the training parameters, creates PyTorch `DataLoader` objects for the training and validation sets, and defines the training loop. The model is trained using the Adam optimizer and a custom balanced cross-entropy loss function.

- **`apply_pytorch.py`**: This script applies the trained model to new seismic data for fault prediction. It loads a pre-trained model from a `.pth` file and performs inference on the specified dataset.

- **`unet3_pytorch.py`**: This file contains the implementation of the 3D U-Net architecture using PyTorch's `nn.Module`. The model structure is equivalent to the original Keras implementation.

- **`utils_pytorch.py`**: This file provides utility functions for data handling, including a `DataGenerator` class that inherits from `torch.utils.data.Dataset` to efficiently load and preprocess data for training.

## Training with Original Parameters

The model was trained using the same parameters as the original implementation by Wu et al. (2019) to ensure a fair comparison:

- **`batch_size`**: 1 (with data augmentation, effectively 2)
- **`dim`**: (128, 128, 128)
- **`n_channels`**: 1
- **`optimizer`**: Adam with a learning rate of `1e-4`
- **`loss`**: Balanced cross-entropy (`cross_entropy_balanced`)

By using the same parameters, we can directly compare the performance of the PyTorch implementation with the original Keras version and the other implementations in this repository.
