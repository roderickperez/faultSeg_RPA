# unet3.py
# Simplified unet for fault segmentation, updated for TensorFlow 2.x

import numpy as np
import os
import tensorflow as tf

# Updated imports from tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate
from tensorflow.keras import backend as K

def unet(pretrained_weights=None, input_size=(None, None, None, 1)):
    """
    Defines the 3D U-Net model architecture using the Keras Functional API.
    This structure is identical between standalone Keras and tf.keras.
    """
    inputs = Input(input_size)
    
    # Encoder Path
    conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    # Bottleneck
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)

    # Decoder Path
    up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv7)

    # Output Layer
    conv8 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])
    
    if pretrained_weights:
        model.load_weights(pretrained_weights)
        
    # model.summary() is called in train.py after instantiation
    return model


def cross_entropy_balanced(y_true, y_pred):
    """
    Custom balanced cross-entropy loss function.
    The logic is based on core TensorFlow operations and remains compatible.
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, 
    # Keras/TF-Keras model with sigmoid activation outputs probabilities.
    # 1. Transform y_pred back to logits
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred = tf.math.log(y_pred / (1 - y_pred))

    # 2. Cast y_true to float
    y_true = tf.cast(y_true, tf.float32)

    # 3. Calculate the balancing weight
    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)
    
    # Matched Keras version by removing epsilon
    beta = count_neg / (count_neg + count_pos)
    pos_weight = beta / (1 - beta)

    # 4. Calculate weighted cross-entropy
    cost = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=pos_weight)

    # 5. Final cost
    cost = tf.reduce_mean(cost * (1 - beta))
    
    # Return 0.0 if there are no positive examples to avoid NaN
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def _to_tensor(x, dtype):
    """
    Convert the input `x` to a tensor of type `dtype`. This helper is fine as is.
    """
    return tf.convert_to_tensor(x, dtype=dtype)