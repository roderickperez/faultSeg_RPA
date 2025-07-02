import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from utils import DataGenerator
from unet3_tensorflow import unet

def main():
    parser = argparse.ArgumentParser(description='Train a U-Net model for fault segmentation.')
    parser.add_argument('--framework', type=str, default='tensorflow', choices=['tensorflow', 'pytorch'], help='The framework to use for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='The initial learning rate.')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size for training.')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs to train for.')
    args = parser.parse_args()

    if args.framework == 'tensorflow':
        train_tensorflow(args)
    else:
        print(f"Framework '{args.framework}' is not supported yet.")

def train_tensorflow(args):
    # Set random seeds for reproducibility
    np.random.seed(12345)
    tf.random.set_seed(1234)

    # Define data paths
    seism_path_train = "./data/train/seis/"
    fault_path_train = "./data/train/fault/"
    seism_path_val = "./data/validation/seis/"
    fault_path_val = "./data/validation/fault/"

    # Get the list of training and validation files
    train_files = os.listdir(seism_path_train)
    valid_files = os.listdir(seism_path_val)

    # Create data generators
    params = {
        'batch_size': args.batch_size,
        'dim': (128, 128, 128),
        'n_channels': 1,
        'shuffle': True
    }
    train_generator = DataGenerator(dpath=seism_path_train, fpath=fault_path_train, data_IDs=range(len(train_files)), **params)
    valid_generator = DataGenerator(dpath=seism_path_val, fpath=fault_path_val, data_IDs=range(len(valid_files)), **params)

    # Build and compile the model
    model = unet(input_size=(None, None, None, 1))
    model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Create callbacks
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    checkpoint_path = "checkpoints/fseg-{epoch:02d}.keras"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')

    if not os.path.exists('logs'):
        os.makedirs('logs')
    logging = TensorBoard(log_dir='./logs')

    callbacks_list = [checkpoint, logging]

    print("Data prepared, ready to train!")

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=args.epochs,
        callbacks=callbacks_list,
        verbose=1
    )

    # Save the final model
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H")
    num_pairs = len(train_files)
    model_name = f"pretrained_model_{num_pairs}_{date_time}.keras"
    model_dir = "./model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, model_name)
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Plot training history
    show_history(history)

def show_history(history):
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.legend(['train', 'test'], loc='center right', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show()

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.legend(['train', 'test'], loc='center right', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show()

if __name__ == '__main__':
    main()
