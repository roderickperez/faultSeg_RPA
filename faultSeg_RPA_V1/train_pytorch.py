import numpy as np
import glob, os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from utils_pytorch import DataGenerator
from unet3_pytorch import * #unet, cross_entropy_balanced

# Set seed for reproducibility
torch.manual_seed(1234)
np.random.seed(12345)

def main():
    os.makedirs("check1", exist_ok=True)
    os.makedirs("log1", exist_ok=True)
    goTrain()

def goTrain():
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    params = {'dim': (128, 128, 128)}
              
    seismPathT = "./data/train/seis/"
    faultPathT = "./data/train/fault/"
    seismPathV = "./data/validation/seis/"
    faultPathV = "./data/validation/fault/"

    # Use a list to allow for shuffling
    # train_ID = list(range(200))
    # valid_ID = list(range(20))
    DATA_DIR = os.path.join(os.getcwd(), "generateSynthData", "data")
    seismPathT = os.path.join(DATA_DIR, "train", "seis")
    faultPathT = os.path.join(DATA_DIR, "train", "fault")
    seismPathV = os.path.join(DATA_DIR, "validation", "seis")
    faultPathV = os.path.join(DATA_DIR, "validation", "fault")

    # discover IDs (strip ".npy")

    train_ID = sorted(int(os.path.splitext(os.path.basename(p))[0])
                    for p in glob.glob(os.path.join(seismPathT, "*.npy")))
    valid_ID = sorted(int(os.path.splitext(os.path.basename(p))[0])
                    for p in glob.glob(os.path.join(seismPathV, "*.npy")))
    print(f"{len(train_ID)} train cubes, {len(valid_ID)} val cubes found.")

    # --- Data Loaders ---
    # Shuffle the list of file IDs before creating the generator to match Keras shuffling
    np.random.shuffle(train_ID)
    train_dataset = DataGenerator(dpath=seismPathT, fpath=faultPathT, data_IDs=train_ID, **params)
    valid_dataset = DataGenerator(dpath=seismPathV, fpath=faultPathV, data_IDs=valid_ID, **params)

    # Effective batch size is 2, matching Keras (original + flipped image)
    # Set shuffle=False to maintain the (original, flipped) pairing from the generator
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    # --- Model, Optimizer, Loss ---
    model = unet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = cross_entropy_balanced # Our custom loss
    
    writer = SummaryWriter('log1') # For TensorBoard logging
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    
    print("Data prepared, ready to train!")

    # --- Training Loop ---
    for epoch in range(100):
        # Training phase
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Accuracy: (predicted_class == true_class)
            acc = (outputs.round() == labels).float().mean()
            running_acc += acc.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader)
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                acc = (outputs.round() == labels).float().mean()
                val_acc += acc.item()
        
        val_loss /= len(valid_loader)
        val_acc /= len(valid_loader)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)

        print(f"Epoch {epoch+1}/{100} - "
              f"Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - "
              f"Val_Loss: {val_loss:.4f} - Val_Acc: {val_acc:.4f}")

        # Save checkpoint (using .pth for PyTorch convention)
        filepath = f"check1/fseg-{epoch+1:02d}.pth"
        torch.save(model.state_dict(), filepath)

    # Save final model
    torch.save(model.state_dict(), 'check1/fseg_final.pth')
    writer.close()
    
    # Create history object for plotting function
    class History:
        def __init__(self, d):
            self.history = d
            
    showHistory(History(history))


def showHistory(history):
    # This function is compatible as it just reads a dictionary
    print(history.history.keys())
    fig = plt.figure(figsize=(10, 6))

    # Summarize history for accuracy
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
    plt.savefig("log1/accuracy_history_pytorch.png")
    plt.show()

    # Summarize history for loss
    fig = plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.legend(['train', 'validation'], loc='center right', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig("log1/loss_history_pytorch.png")
    plt.show()


if __name__ == '__main__':
    main()