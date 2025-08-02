# utils_keras.py (Corrected)

import numpy as np
import tensorflow as tf # Use tensorflow.keras
import random

class DataGenerator(tf.keras.utils.Sequence):
  'Generates data for Keras'
  def __init__(self, dpath, fpath, data_IDs, batch_size=1, dim=(128,128,128),
               n_channels=1, shuffle=True):
    'Initialization'
    self.dim = dim
    self.dpath = dpath
    self.fpath = fpath
    self.batch_size = batch_size
    self.data_IDs = data_IDs
    self.n_channels = n_channels
    self.shuffle = shuffle
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.data_IDs) / self.batch_size))

  def __getitem__(self, index):
    'Generates one batch of data'
    # Generate indexes of the batch
    indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

    # Find list of IDs
    data_IDs_temp = [self.data_IDs[k] for k in indexes]

    # Generate data
    X, Y = self.__data_generation(data_IDs_temp)

    return X, Y

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.data_IDs))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, data_IDs_temp):
    'Generates data containing batch_size samples'
    # Initialize lists to hold the batch data
    X_list = []
    Y_list = []

    # --- FIX ---
    # Loop through each ID in the batch. The original code only used data_IDs_temp[0].
    for data_id in data_IDs_temp:
        # Load data for the current ID
        gx = np.fromfile(self.dpath + str(data_id) + '.dat', dtype=np.single)
        fx = np.fromfile(self.fpath + str(data_id) + '.dat', dtype=np.single)
        
        gx = np.reshape(gx, self.dim)
        fx = np.reshape(fx, self.dim)
        
        # Standardize seismic data
        xm = np.mean(gx)
        xs = np.std(gx)
        gx = (gx - xm) / xs
        
        # Transpose dimensions as in the original code
        # from (n1,n2,n3) to (n3,n2,n1)
        gx = np.transpose(gx)
        fx = np.transpose(fx)
        
        # --- AUGMENTATION ---
        # The original code hard-coded 2 augmentations. We do the same
        # for each sample in the batch.
        
        # Augmentation 1: Original
        X_list.append(np.reshape(gx, (*self.dim, self.n_channels)))
        Y_list.append(np.reshape(fx, (*self.dim, self.n_channels)))
        
        # Augmentation 2: Flipped Up-Down
        # This augmentation is applied identically to the image and the mask
        X_list.append(np.reshape(np.flipud(gx), (*self.dim, self.n_channels)))
        Y_list.append(np.reshape(np.flipud(fx), (*self.dim, self.n_channels)))

    # Convert lists to a single numpy array.
    # The final batch size will be 2 * self.batch_size due to augmentation.
    return np.array(X_list, dtype=np.single), np.array(Y_list, dtype=np.single)