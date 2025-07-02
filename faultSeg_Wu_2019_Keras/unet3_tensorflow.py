
# Simplified unet for fault segmentation
# The original u-net architecture is more complicated than necessary 
# for our task of fault segmentation.
# We significanlty reduce the number of layers and features at each 
# layer to save GPU memory and computation but still preserve high 
# performace in fault segmentation.

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv3D,
    MaxPooling3D,
    UpSampling3D,
    concatenate,
)

def unet(input_size=(None, None, None, 1)):
    """
    Builds a 3D U-Net model.
    
    Args:
        input_size (tuple): The shape of the input tensor.
        
    Returns:
        A Keras Model instance.
    """
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    # Bridge
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)

    # Decoder
    up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv7)

    # Output layer
    conv8 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])
    
    return model
