import numpy as np
from numpy.random import seed
seed(12345)

import os
import tensorflow as tf
# Updated random seed setting for TF 2.x
tf.random.set_seed(1234)

import matplotlib.pyplot as plt

# Updated imports from tensorflow.keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K

# Local imports (ensure these are also TF 2.x compatible)
from utils_tf import DataGenerator
from unet3_tf import * #unet, cross_entropy_balanced

def main():
    # Set up directories
    os.makedirs("check1", exist_ok=True)
    os.makedirs("log1", exist_ok=True)
    goTrain()

def goTrain():
    # input image dimensions
    params = {'batch_size': 1,  # Set to 1 because generator returns a batch of 2
              'dim': (128, 128, 128),
              'n_channels': 1,
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
    
    # Use 'learning_rate' instead of 'lr' for optimizers in TF 2.x
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=cross_entropy_balanced,
              metrics=['accuracy'])
    model.summary()

    # Checkpoint callback
    filepath = "check1/fseg-{epoch:02d}.keras"
    # Changed to 'val_accuracy' to match TF 2.x history key
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy',
                                 verbose=1, save_best_only=False, mode='max')
    
    # Updated TensorBoard callback for TF 2.x
    logging = TrainValTensorBoard(log_dir='./log1')
    
    callbacks_list = [checkpoint, logging]
    print("Data prepared, ready to train!")
    
    # Use model.fit() which replaces model.fit_generator() in TF 2.x
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=100,
        callbacks=callbacks_list,
        verbose=1
    )
    
    model.save('check1/fseg_final.keras')
    showHistory(history)


def showHistory(history):
    # list all data in history
    print(history.history.keys())
    fig = plt.figure(figsize=(10, 6))

    # summarize history for accuracy
    # Changed to 'accuracy' and 'val_accuracy' to match history keys
    # ----- summarize history for accuracy (works with either key name) -----
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    val_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'

    plt.plot(history.history[acc_key])
    plt.plot(history.history[val_key])

    plt.title('Model accuracy', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.legend(['train', 'validation'], loc='center right', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig("log1/accuracy_history.png")
    plt.show()

    # summarize history for loss
    fig = plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.legend(['train', 'validation'], loc='center right', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig("log1/loss_history.png")
    plt.show()

# Updated custom TensorBoard callback for TensorFlow 2.x
class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./log1', **kwargs):
        self.val_log_dir = os.path.join(log_dir, 'validation')
        # Call the parent constructor with the training log directory
        super(TrainValTensorBoard, self).__init__(log_dir, **kwargs)

    def set_model(self, model):
        # Create a file writer for validation logs
        self.val_writer = tf.summary.create_file_writer(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_logs = {k: v for k, v in logs.items() if k.startswith('val_')}
        
        # Use the validation writer to write validation logs
        with self.val_writer.as_default():
            for name, value in val_logs.items():
                # TF 2.x summary API
                tf.summary.scalar(name.replace('val_', ''), value, step=epoch)
        self.val_writer.flush()
        
        # Pass the training logs to the parent class for writing
        train_logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        
        # Also log learning rate
        lr = K.get_value(self.model.optimizer.learning_rate)
        train_logs.update({'learning_rate': lr})

        super(TrainValTensorBoard, self).on_epoch_end(epoch, train_logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


if __name__ == '__main__':
    # Optional: Configure GPU memory growth to avoid CUDNN errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Running on {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(e)
    
    main()