from numpy.random import seed
seed(12345)
from tensorflow import set_random_seed
set_random_seed(1234)
import os
import random
import numpy as np
import skimage
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard
from tensorflow.keras import backend as keras
from utils_keras import DataGenerator
from unet3_keras import * # unet, cross_entropy_balanced

from tensorflow.keras.utils import get_custom_objects
get_custom_objects()['cross_entropy_balanced'] = cross_entropy_balanced

def main():
  goTrain()

def goTrain():
  # input image dimensions
  params = {'batch_size':1, # Originally 1, increased for data augmentation
          'dim':(128,128,128),
          'n_channels':1,
          'shuffle': True}
  seismPathT = "./data/train/seis/"
  faultPathT = "./data/train/fault/"

  seismPathV = "./data/validation/seis/"
  faultPathV = "./data/validation/fault/"
  train_ID = range(200)
  valid_ID = range(20)
  train_generator = DataGenerator(dpath=seismPathT,fpath=faultPathT,
                                  data_IDs=train_ID,**params)
  valid_generator = DataGenerator(dpath=seismPathV,fpath=faultPathV,
                                  data_IDs=valid_ID,**params)
  model = unet(input_size=(None, None, None,1))
  # model.compile(optimizer=Adam(lr=1e-4), loss=binary_crossentropy, 
  #               metrics=['accuracy'])
  model.compile(optimizer=Adam(lr=1e-4), loss=cross_entropy_balanced, 
              metrics=['accuracy'])
  model.summary()

  # checkpoint
  filepath = "check1/fseg-{epoch:02d}.keras"
  checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', 
        verbose=1, save_best_only=False, mode='max')
  logging = TrainValTensorBoard()
  #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, 
  #                              patience=20, min_lr=1e-8)
  callbacks_list = [checkpoint, logging]
  print("data prepared, ready to train!")
  # Fit the model
  history = model.fit(
        train_generator,                     # first positional arg is the data
        validation_data=valid_generator,
        epochs=100,
        callbacks=callbacks_list,
        verbose=1)
  model.save('check1/fseg.keras')
  showHistory(history)

def showHistory(history):
  # list all data in history
  print(history.history.keys())
  fig = plt.figure(figsize=(10,6))

  # summarize history for accuracy
  # ----- summarize history for accuracy (works with either key name) -----
  acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
  val_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'

  plt.plot(history.history[acc_key])
  plt.plot(history.history[val_key])
  
  plt.title('Model accuracy',fontsize=20)
  plt.ylabel('Accuracy',fontsize=20)
  plt.xlabel('Epoch',fontsize=20)
  plt.legend(['train', 'test'], loc='center right',fontsize=20)
  plt.tick_params(axis='both', which='major', labelsize=18)
  plt.tick_params(axis='both', which='minor', labelsize=18)
  plt.show()

  # summarize history for loss
  fig = plt.figure(figsize=(10,6))
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss',fontsize=20)
  plt.ylabel('Loss',fontsize=20)
  plt.xlabel('Epoch',fontsize=20)
  plt.legend(['train', 'test'], loc='center right',fontsize=20)
  plt.tick_params(axis='both', which='major', labelsize=18)
  plt.tick_params(axis='both', which='minor', labelsize=18)
  plt.show()

# Replace the old TrainValTensorBoard class in train_keras.py with this modern version:

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./log1', **kwargs):
        self.val_log_dir = os.path.join(log_dir, 'validation')
        # Call the parent constructor with the training log directory
        super(TrainValTensorBoard, self).__init__(log_dir, **kwargs)

    def set_model(self, model):
        # Create a file writer for validation logs using the modern TF 2.x API
        self.val_writer = tf.summary.create_file_writer(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_logs = {k: v for k, v in logs.items() if k.startswith('val_')}
        
        # Use the validation writer to write validation logs
        with self.val_writer.as_default():
            for name, value in val_logs.items():
                # Use the modern TF 2.x summary API
                tf.summary.scalar(name.replace('val_', ''), value, step=epoch)
        self.val_writer.flush()
        
        # Pass only the training logs to the parent class
        train_logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, train_logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

if __name__ == '__main__':
    main()
