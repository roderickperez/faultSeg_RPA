import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from unet3_pytorch import * #unet

# --- Configuration ---
pngDir = './png/'
os.makedirs(pngDir, exist_ok=True)
os.makedirs(os.path.join(pngDir, 'f3d'), exist_ok=True)

# --- Setup Device and Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Where is this script running from?
try:
    ROOT_DIR = Path(__file__).resolve().parent  # if apply_torch.py lives at repo root
except NameError:  # e.g., in notebooks
    ROOT_DIR = Path.cwd()


DATA_DIR = os.path.join(ROOT_DIR, "generateSynthData", "data")
seismPathV = os.path.join(DATA_DIR, "validation", "seis")
faultPathV = os.path.join(DATA_DIR, "validation", "fault")

# also, make sure the model path matches where you saved weights:
model_path = 'check1/fseg-70.pth'  # or fseg_final.pth

model = unet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval() # Set the model to evaluation mode

def main():
    # goTrainTest()
    goValidTest()
    goF3Test()

def predict(gx_numpy, n1, n2, n3):
    """Helper function to run prediction on a numpy array."""
    # Convert numpy array to torch tensor, add batch and channel dims, and move to device
    # Shape becomes [1, 1, n1, n2, n3]
    gx_tensor = torch.from_numpy(gx_numpy).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float)
    
    with torch.no_grad():
        fp_tensor = model(gx_tensor)
        
    # Move tensor to CPU, remove batch/channel dims, and convert to numpy
    # Shape becomes [n1, n2, n3]
    fp_numpy = fp_tensor.squeeze().cpu().numpy()
    return fp_numpy

def goTrainTest():
    # ... (code identical to goValidTest but with different paths/IDs)
    pass

def goValidTest():
    seismPath = seismPathV + '/'
    faultPath = faultPathV + '/'
    n1, n2, n3 = 128, 128, 128
    dk = 0  # pick a valid id you actually have

    # >>> load NPY, not DAT
    gx = np.load(seismPath + str(dk) + '.npy').astype(np.single)
    fx = np.load(faultPath + str(dk) + '.npy').astype(np.single)

    gx = np.reshape(gx, (n1, n2, n3))
    fx = np.reshape(fx, (n1, n2, n3))

    gm, gs = np.mean(gx), np.std(gx)
    gx = (gx - gm) / gs
    gx = np.transpose(gx)
    fx = np.transpose(fx)

    fp = predict(gx, n1, n2, n3)

    # Slicing for visualization
    gx1, fx1, fp1 = gx[50, :, :], fx[50, :, :], fp[50, :, :]
    gx2, fx2, fp2 = gx[:, 29, :], fx[:, 29, :], fp[:, 29, :]
    gx3, fx3, fp3 = gx[:, :, 29], fx[:, :, 29], fp[:, :, 29]
    
    plot2d(gx1, fx1, fp1, png='fp1_pytorch')
    plot2d(gx2, fx2, fp2, png='fp2_pytorch')
    plot2d(gx3, fx3, fp3, png='fp3_pytorch')

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
    fp = predict(gx, n1, n2, n3)

    # Slicing for visualization
    gx1, fp1 = gx[99, :, :], fp[99, :, :]
    gx2, fp2 = gx[:, 29, :], fp[:, 29, :]
    gx3, fp3 = gx[:, :, 29], fp[:, :, 29]
    
    # Note: fx is not available for f3d, so passing fp as a placeholder
    plot2d(gx1, fp1, fp1, at=1, png='f3d/fp1_pytorch')
    plot2d(gx2, fp2, fp2, at=2, png='f3d/fp2_pytorch')
    plot2d(gx3, fp3, fp3, at=2, png='f3d/fp3_pytorch')

def plot2d(gx, fx, fp, at=1, png=None):
    # This function is unchanged as it works with NumPy arrays
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(131)
    ax.set_title("Seismic Image")
    ax.imshow(gx, vmin=-2, vmax=2, cmap=plt.cm.bone, interpolation='bicubic', aspect=at)
    ax = fig.add_subplot(132)
    ax.set_title("True Faults")
    ax.imshow(fx, vmin=0, vmax=1, cmap=plt.cm.bone, interpolation='bicubic', aspect=at)
    ax = fig.add_subplot(133)
    ax.set_title("Predicted Faults")
    ax.imshow(fp, vmin=0, vmax=1.0, cmap=plt.cm.bone, interpolation='bicubic', aspect=at)
    if png:
        plt.savefig(os.path.join(pngDir, png + '.png'))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()