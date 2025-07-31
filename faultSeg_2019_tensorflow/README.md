# FaultSeg: TensorFlow Implementation

This directory contains a TensorFlow implementation of the U-Net model for seismic fault segmentation, based on the original work by Wu et al. (2019). The code has been adapted to use the modern TensorFlow 2.x API with `tf.keras`.

## Code Description

The TensorFlow implementation is organized into the following files:

- **`train_tf.py`**: This script is used to train the U-Net model. It sets up the training parameters, creates `tf.data.Dataset` objects for both training and validation data, and compiles the model using the Adam optimizer and a balanced cross-entropy loss function.

- **`apply_tf.py`**: This script applies the trained model to new seismic data for fault prediction. It loads a pre-trained model and performs inference on the specified dataset.

- **`unet3_tf.py`**: This file contains the implementation of the 3D U-Net architecture using the `tf.keras` API. The model structure is identical to the original Keras implementation.

- **`utils_tf.py`**: This file provides utility functions for data handling, including a `DataGenerator` class that is compatible with `tf.keras` and can be used to efficiently load and preprocess data for training.

## Training with Original Parameters

The model was trained using the same parameters as the original implementation by Wu et al. (2019) to ensure a fair comparison:

- **`batch_size`**: 1
- **`dim`**: (128, 128, 128)
- **`n_channels`**: 1
- **`optimizer`**: Adam with a learning rate of `1e-4`
- **`loss`**: Balanced cross-entropy (`cross_entropy_balanced`)

By using the same parameters, we can directly compare the performance of the TensorFlow implementation with the original Keras version and the other implementations in this repository.
