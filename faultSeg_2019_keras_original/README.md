# FaultSeg: Original Keras Implementation

This directory contains the original Keras implementation of the U-Net model for seismic fault segmentation, as described in the paper by Wu et al. (2019).

## Code Description

The code is organized into the following files:

- **`train.py`**: This script is used to train the U-Net model. It defines the training parameters, loads the training and validation data using a `DataGenerator`, compiles the model with an Adam optimizer and binary cross-entropy loss, and fits the model to the data.

- **`apply.py`**: This script applies the trained model to new seismic data for fault prediction. It loads a pre-trained model from a checkpoint file and performs inference on training, validation, or test datasets.

- **`unet3.py`**: This file contains the implementation of the 3D U-Net architecture. The model consists of an encoder-decoder structure with skip connections, designed to capture both contextual and localized features in the seismic data.

- **`utils.py`**: This file provides utility functions for data handling, including a `DataGenerator` class that loads and preprocesses seismic and fault data in batches.

## Parameters and Usage

### Training

To train the model, you can run the `train.py` script. The key parameters used in the training process are:

- **`batch_size`**: 1
- **`dim`**: (128, 128, 128)
- **`n_channels`**: 1
- **`optimizer`**: Adam with a learning rate of `1e-4`
- **`loss`**: Binary cross-entropy

### Prediction

To apply the trained model for prediction, you can use the `apply.py` script. This script loads a trained model from a `.hdf5` file and applies it to the seismic data to generate fault predictions.
