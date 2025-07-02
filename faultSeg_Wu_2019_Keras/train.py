from numpy.random import seed
seed(12345)
from tensorflow.random import set_seed
set_seed(1234)
import os
import random
import numpy as np
import skimage
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard
from tensorflow.keras import backend as K
from utils import DataGenerator
from unet3 import *
from datetime import datetime

def main():
  goTrain()

def goTrain():
  # input image dimensions
  params = {'batch_size':1,
          'dim':(128,128,128),
          'n_channels':1,
          'shuffle': True}
  seismPathT = "./data/train/seis/"
  faultPathT = "./data/train/fault/"

  seismPathV = "./data/validation/seis/"
  faultPathV = "./data/validation/fault/"
  
  train_files = os.listdir(seismPathT)
  valid_files = os.listdir(seismPathV)
  
  train_ID = range(len(train_files))
  valid_ID = range(len(valid_files))
  train_generator = DataGenerator(dpath=seismPathT,fpath=faultPathT,
                                  data_IDs=train_ID,**params)
  valid_generator = DataGenerator(dpath=seismPathV,fpath=faultPathV,
                                  data_IDs=valid_ID,**params)
  model = unet(input_size=(None, None, None,1))
  model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', 
                metrics=['accuracy'])
  model.summary()

  # checkpoint
  if not os.path.exists('check1'):
    os.makedirs('check1')
  filepath="check1/fseg-{epoch:02d}.keras"
  checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', 
        verbose=1, save_best_only=False, mode='max')
  if not os.path.exists('log1'):
    os.makedirs('log1')
  logging = TensorBoard(log_dir='./log1')
  #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, 
  #                              patience=20, min_lr=1e-8)
  callbacks_list = [checkpoint, logging]
  print("data prepared, ready to train!")
  # Fit the model
  history=model.fit(train_generator,
  validation_data=valid_generator,epochs=100,callbacks=callbacks_list,verbose=1)
  
  now = datetime.now()
  date_time = now.strftime("%Y-%m-%d_%H")
  num_pairs = len(train_ID)
  model_name = f"pretrained_model_{num_pairs}_{date_time}.keras"
  model_dir = "/Users/roderickperez/Documents/DS_Projects/faultSegm/faultSeg_Wu_2019_Keras/model"
  if not os.path.exists(model_dir):
      os.makedirs(model_dir)
  model_path = os.path.join(model_dir, model_name)
  model.save(model_path)
  print(f"Model saved to {model_path}")

  showHistory(history)

def showHistory(history):
  # list all data in history
  print(history.history.keys())
  fig = plt.figure(figsize=(10,6))

  # summarize history for accuracy
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
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

if __name__ == '__main__':
    main()