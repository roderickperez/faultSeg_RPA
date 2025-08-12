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
    # Initialization
    # We only use the first ID, as the augmentations will create the full batch.
    gx  = np.fromfile(self.dpath+str(data_IDs_temp[0])+'.dat',dtype=np.single)
    fx  = np.fromfile(self.fpath+str(data_IDs_temp[0])+'.dat',dtype=np.single)
    gx = np.reshape(gx,self.dim)
    fx = np.reshape(fx,self.dim)

    # Standardize seismic data
    xm = np.mean(gx)
    xs = np.std(gx)
    gx = (gx - xm) / xs

    # Transpose dimensions
    gx = np.transpose(gx)
    fx = np.transpose(fx)

    # Generate augmented data (4 rotations)
    X = np.zeros((4, *self.dim, self.n_channels), dtype=np.single)
    Y = np.zeros((4, *self.dim, self.n_channels), dtype=np.single)
    for i in range(4):
        X[i,] = np.reshape(np.rot90(gx, i, (1, 2)), (*self.dim, self.n_channels))
        Y[i,] = np.reshape(np.rot90(fx, i, (1, 2)), (*self.dim, self.n_channels))
    
    return X, Y