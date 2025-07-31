import numpy as np
# Changed from 'import keras' to 'import tensorflow as tf'
import tensorflow as tf

# The DataGenerator now inherits from tf.keras.utils.Sequence
class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for a Keras/TF-Keras model'
    def __init__(self, dpath, fpath, data_IDs, batch_size=1, dim=(128, 128, 128),
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
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, data_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization (replicates Keras version which effectively ignores batch_size > 1)
        X = np.zeros((2, *self.dim, self.n_channels), dtype=np.single)
        Y = np.zeros((2, *self.dim, self.n_channels), dtype=np.single)
        
        # Load data for the first ID in the batch temp list
        gx = np.fromfile(self.dpath + str(data_IDs_temp[0]) + '.dat', dtype=np.single)
        fx = np.fromfile(self.fpath + str(data_IDs_temp[0]) + '.dat', dtype=np.single)
        gx = np.reshape(gx, self.dim)
        fx = np.reshape(fx, self.dim)
        
        # Normalize (matched Keras version by removing epsilon)
        xm = np.mean(gx)
        xs = np.std(gx)
        gx = (gx - xm) / xs
        
        # Transpose
        gx = np.transpose(gx)
        fx = np.transpose(fx)

        # Generate augmented data
        X[0, ] = np.reshape(gx, (*self.dim, self.n_channels))
        Y[0, ] = np.reshape(fx, (*self.dim, self.n_channels))
        X[1, ] = np.reshape(np.flipud(gx), (*self.dim, self.n_channels))
        Y[1, ] = np.reshape(np.flipud(fx), (*self.dim, self.n_channels))

        return X, Y