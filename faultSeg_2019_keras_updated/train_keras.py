# train_keras.py (Corrected and Improved)

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from utils_keras import DataGenerator # Use the corrected version
from unet3_keras import unet, cross_entropy_balanced

# --- FIX 1: Modern seed setting ---
np.random.seed(12345)
tf.random.set_seed(1234)

# Ensure the custom loss is registered with Keras
tf.keras.utils.get_custom_objects()['cross_entropy_balanced'] = cross_entropy_balanced

def main():
  goTrain()

def goTrain():
  # input image dimensions
  params = {'batch_size': 2, # Can be increased for better performance
            'dim':(128,128,128),
            'n_channels':1,
            'shuffle': True}
  
  seismPathT = "./data/train/seis/"
  faultPathT = "./data/train/fault/"
  seismPathV = "./data/validation/seis/"
  faultPathV = "./data/validation/fault/"
  
  train_ID = range(200)
  valid_ID = range(20)
  
  train_generator = DataGenerator(dpath=seismPathT, fpath=faultPathT,
                                  data_IDs=train_ID, **params)
  valid_generator = DataGenerator(dpath=seismPathV, fpath=faultPathV,
                                  data_IDs=valid_ID, **params)
                                  
  model = unet(input_size=(None, None, None, 1))
  model.compile(optimizer=Adam(learning_rate=1e-4), loss=cross_entropy_balanced,
                metrics=['accuracy'])
  model.summary()

  # input image dimensions
  params = {'batch_size': 1, # Set to 1, as the generator creates a batch of 4 from one sample
            'dim':(128,128,128),
            'n_channels':1,
            'shuffle': True}
  
  seismPathT = "./data/train/seis/"
  faultPathT = "./data/train/fault/"
  seismPathV = "./data/validation/seis/"
  faultPathV = "./data/validation/fault/"
  
  train_ID = range(200)
  valid_ID = range(20)
  
  train_generator = DataGenerator(dpath=seismPathT, fpath=faultPathT,
                                  data_IDs=train_ID, **params)
  valid_generator = DataGenerator(dpath=seismPathV, fpath=faultPathV,
                                  data_IDs=valid_ID, **params)
                                  
  model = unet(input_size=(None, None, None, 1))
  model.compile(optimizer=Adam(learning_rate=1e-4), loss=cross_entropy_balanced,
                metrics=['accuracy'])
  model.summary()

  # --- FIX 2: Consistent checkpoint path ---
  checkpoint_dir = "checkpoints"
  os.makedirs(checkpoint_dir, exist_ok=True)
  filepath = os.path.join(checkpoint_dir, "fseg-{epoch:02d}.keras")
  
  checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy',
                               verbose=1, save_best_only=False, mode='max')
  
  # --- FIX 3: Simplified TensorBoard callback ---
  # The standard callback handles validation logs automatically.
  tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
  
  callbacks_list = [checkpoint, tensorboard_callback]
  
  print("Data prepared, ready to train!")
  
  # Fit the model
  history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=25, # Corrected epochs to match paper
        callbacks=callbacks_list,
        verbose=1)
  
  # Save the final model
  model.save(os.path.join(checkpoint_dir, 'fseg_final.keras'))
  
  showHistory(history)

def showHistory(history):
  # (Your showHistory function is fine, no changes needed)
  print(history.history.keys())
  fig = plt.figure(figsize=(10,6))
  acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
  val_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
  plt.plot(history.history[acc_key])
  plt.plot(history.history[val_key])
  plt.title('Model accuracy',fontsize=20)
  plt.ylabel('Accuracy',fontsize=20)
  plt.xlabel('Epoch',fontsize=20)
  plt.legend(['train', 'test'], loc='center right',fontsize=20)
  plt.tick_params(axis='both', which='major', labelsize=18)
  plt.show()

  fig = plt.figure(figsize=(10,6))
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss',fontsize=20)
  plt.ylabel('Loss',fontsize=20)
  plt.xlabel('Epoch',fontsize=20)
  plt.legend(['train', 'test'], loc='center right',fontsize=20)
  plt.tick_params(axis='both', which='major', labelsize=18)
  plt.show()

if __name__ == '__main__':
    main()