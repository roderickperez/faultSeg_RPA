# FaultSeg: Updated Keras Implementation

This directory contains an updated and refactored version of the original Keras code for seismic fault segmentation. The code has been modernized to be compatible with the latest versions of TensorFlow and Keras.

## Code Description

The updated implementation includes the following files:

- **`train_keras.py`**: This script is used to train the U-Net model. It has been updated to use the `tensorflow.keras` API and includes a more modern `TrainValTensorBoard` callback for logging training and validation metrics.

- **`apply_keras.py`**: This script applies the trained model to new seismic data for fault prediction. It has been updated to load models saved in the `.keras` format.

- **`unet3_keras.py`**: This file contains the implementation of the 3D U-Net architecture, updated to use the `tensorflow.keras` API.

- **`utils_keras.py`**: This file provides utility functions for data handling, including a `DataGenerator` class that is compatible with the `tensorflow.keras` API.

## Training with Original Parameters

The model was trained using the same parameters as the original implementation by Wu et al. (2019) to ensure a fair comparison:

- **`batch_size`**: 1
- **`dim`**: (128, 128, 128)
- **`n_channels`**: 1
- **`optimizer`**: Adam with a learning rate of `1e-4`
- **`loss`**: Balanced cross-entropy (`cross_entropy_balanced`)

By using the same parameters, we can directly compare the performance of the updated implementation with the original one, as well as with the TensorFlow and PyTorch versions.
