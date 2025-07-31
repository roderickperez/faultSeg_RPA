import math
import skimage
import numpy as np
import os
import matplotlib.pyplot as plt

# Changed to import from tensorflow.keras
import tensorflow as tf
from tensorflow.keras.models import load_model

# Ensure the custom loss function from the updated unet3.py is available
from unet3_tf import * # cross_entropy_balanced

# --- Configuration ---
pngDir = './png/'
os.makedirs(pngDir, exist_ok=True)
os.makedirs(os.path.join(pngDir, 'f3d'), exist_ok=True)

# Load the Keras model. TF 2.x can load HDF5 models saved from older versions.
# The custom_objects dictionary is crucial for loading models with custom loss functions.
model = load_model('check/fseg-70.keras',
                   custom_objects={
                       'cross_entropy_balanced': cross_entropy_balanced
                   })

def main():
    # goTrainTest()
    goValidTest()
    goF3Test()

def goTrainTest():
    seismPath = "./data/train/seis/"
    faultPath = "./data/train/fault/"
    n1, n2, n3 = 128, 128, 128
    dk = 100
    gx = np.fromfile(seismPath + str(dk) + '.dat', dtype=np.single)
    fx = np.fromfile(faultPath + str(dk) + '.dat', dtype=np.single)
    gx = np.reshape(gx, (n1, n2, n3))
    fx = np.reshape(fx, (n1, n2, n3))
    
    # Pre-processing
    gm = np.mean(gx)
    gs = np.std(gx)
    gx = (gx - gm) / gs
    gx = np.transpose(gx)
    fx = np.transpose(fx)

    # Prediction
    fp = model.predict(np.reshape(gx, (1, n1, n2, n3, 1)), verbose=1)
    fp = fp[0, :, :, :, 0]

    # Slicing for visualization
    gx1 = gx[50, :, :]
    fx1 = fx[50, :, :]
    fp1 = fp[50, :, :]
    plot2d(gx1, fx1, fp1, png='fp')

def goValidTest():
    seismPath = "./data/validation/seis/"
    faultPath = "./data/validation/fault/"
    n1, n2, n3 = 128, 128, 128
    dk = 2
    gx = np.fromfile(seismPath + str(dk) + '.dat', dtype=np.single)
    fx = np.fromfile(faultPath + str(dk) + '.dat', dtype=np.single)
    gx = np.reshape(gx, (n1, n2, n3))
    fx = np.reshape(fx, (n1, n2, n3))
    
    # Pre-processing
    gm = np.mean(gx)
    gs = np.std(gx)
    gx = (gx - gm) / gs
    gx = np.transpose(gx)
    fx = np.transpose(fx)

    # Prediction
    fp = model.predict(np.reshape(gx, (1, n1, n2, n3, 1)), verbose=1)
    fp = fp[0, :, :, :, 0]

    # Slicing for visualization
    gx1 = gx[50, :, :]
    fx1 = fx[50, :, :]
    fp1 = fp[50, :, :]
    gx2 = gx[:, 29, :]
    fx2 = fx[:, 29, :]
    fp2 = fp[:, 29, :]
    gx3 = gx[:, :, 29]
    fx3 = fx[:, :, 29]
    fp3 = fp[:, :, 29]
    
    plot2d(gx1, fx1, fp1, png='fp1')
    plot2d(gx2, fx2, fp2, png='fp2')
    plot2d(gx3, fx3, fp3, png='fp3')

def goF3Test():
    seismPath = "./data/prediction/f3d/"
    n3, n2, n1 = 512, 384, 128
    gx = np.fromfile(seismPath + 'gxl.dat', dtype=np.single)
    gx = np.reshape(gx, (n3, n2, n1))
    
    # Pre-processing
    gm = np.mean(gx)
    gs = np.std(gx)
    gx = (gx - gm) / gs
    gx = np.transpose(gx)

    # Prediction
    # Note: The input shape to predict is (n1, n2, n3) after transpose
    fp = model.predict(np.reshape(gx, (1, n1, n2, n3, 1)), verbose=1)
    fp = fp[0, :, :, :, 0]

    # Slicing for visualization
    gx1 = gx[99, :, :]
    fp1 = fp[99, :, :]
    gx2 = gx[:, 29, :]
    fp2 = fp[:, 29, :]
    gx3 = gx[:, :, 29]
    fp3 = fp[:, :, 29]
    
    plot2d(gx1, fp1, fp1, at=1, png='f3d/fp1')
    plot2d(gx2, fp2, fp2, at=2, png='f3d/fp2')
    plot2d(gx3, fp3, fp3, at=2, png='f3d/fp3')

def plot2d(gx, fx, fp, at=1, png=None):
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Seismic data
    ax = fig.add_subplot(131)
    ax.set_title("Seismic Image")
    ax.imshow(gx, vmin=-2, vmax=2, cmap=plt.cm.bone, interpolation='bicubic', aspect=at)
    
    # Plot 2: True Faults
    ax = fig.add_subplot(132)
    ax.set_title("True Faults")
    ax.imshow(fx, vmin=0, vmax=1, cmap=plt.cm.bone, interpolation='bicubic', aspect=at)
    
    # Plot 3: Predicted Faults
    ax = fig.add_subplot(133)
    ax.set_title("Predicted Faults")
    ax.imshow(fp, vmin=0, vmax=1.0, cmap=plt.cm.bone, interpolation='bicubic', aspect=at)
    
    if png:
        plt.savefig(os.path.join(pngDir, png + '.png'))
        
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Ensure a GPU is available and configured
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
            
    main()