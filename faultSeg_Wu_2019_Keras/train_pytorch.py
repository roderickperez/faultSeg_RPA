import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from utils import DataGenerator
from unet3_pytorch import UNet

def main():
    parser = argparse.ArgumentParser(description='Train a U-Net model for fault segmentation.')
    parser.add_argument('--framework', type=str, default='pytorch', choices=['tensorflow', 'pytorch'], help='The framework to use for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='The initial learning rate.')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size for training.')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs to train for.')
    args = parser.parse_args()

    if args.framework == 'pytorch':
        train_pytorch(args)
    else:
        print(f"Framework '{args.framework}' is not supported yet.")

def train_pytorch(args):
    # Set random seeds for reproducibility
    torch.manual_seed(1234)
    np.random.seed(12345)

    # Define data paths
    seism_path_train = "./data/train/seis/"
    fault_path_train = "./data/train/fault/"
    seism_path_val = "./data/validation/seis/"
    fault_path_val = "./data/validation/fault/"

    # Get the list of training and validation files
    train_files = os.listdir(seism_path_train)
    valid_files = os.listdir(seism_path_val)

    # Create data loaders
    params = {
        'batch_size': args.batch_size,
        'dim': (128, 128, 128),
        'n_channels': 1,
        'shuffle': True
    }
    train_loader = DataLoader(DataGenerator(dpath=seism_path_train, fpath=fault_path_train, data_IDs=range(len(train_files)), **params), batch_size=args.batch_size)
    valid_loader = DataLoader(DataGenerator(dpath=seism_path_val, fpath=fault_path_val, data_IDs=range(len(valid_files)), **params), batch_size=args.batch_size)

    # Build the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.nelement()
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.nelement()
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(valid_loader)
        val_acc = 100 * correct / total
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

    # Save the final model
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H")
    num_pairs = len(train_files)
    model_name = f"pretrained_model_{num_pairs}_{date_time}.pth"
    model_dir = "./model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot training history
    show_history(history)

def show_history(history):
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.legend(['train', 'test'], loc='center right', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show()

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.legend(['train', 'test'], loc='center right', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show()

if __name__ == '__main__':
    main()
