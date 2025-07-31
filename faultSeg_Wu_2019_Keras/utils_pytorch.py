import numpy as np
import torch
from torch.utils.data import Dataset
import os

class DataGenerator(Dataset):
    'Generates data for PyTorch'
    def __init__(self, dpath, fpath, data_IDs, dim=(128, 128, 128), n_channels=1):
        'Initialization'
        self.dim = dim
        self.dpath = dpath
        self.fpath = fpath
        self.data_IDs = data_IDs
        self.n_channels = n_channels
        # The original code's augmentation created a batch of 2 from one sample.
        # We replicate this by doubling the dataset length and using the index
        # to decide whether to apply the flip augmentation.
        self.augmented_length = len(self.data_IDs) * 2

    def __len__(self):
        'Denotes the total number of samples'
        return self.augmented_length

    def __getitem__(self, index):
        'Generates one sample of data'
        # Determine original data ID and if augmentation should be applied
        original_index = index // 2
        apply_flip = (index % 2 == 1)
        
        ID = self.data_IDs[original_index]

        # Load data
        gx = np.load(os.path.join(self.dpath, f"{ID}.npy")).astype(np.single)
        fx = np.load(os.path.join(self.fpath, f"{ID}.npy")).astype(np.single)
        
        gx = np.reshape(gx, self.dim)
        fx = np.reshape(fx, self.dim)

        # Normalize seismic data (matched Keras version by removing epsilon)
        xm = np.mean(gx)
        xs = np.std(gx)
        gx = (gx - xm) / xs

        # Transpose to (n1, n2, n3)
        gx = np.transpose(gx)
        fx = np.transpose(fx)
        
        # Apply flip augmentation if required
        if apply_flip:
            gx = np.flipud(gx).copy() # Use .copy() to avoid negative stride issues
            fx = np.flipud(fx).copy()

        # Add channel dimension and convert to PyTorch tensors
        # PyTorch expects (C, D, H, W)
        X = torch.from_numpy(gx).unsqueeze(0) # Shape: [1, 128, 128, 128]
        Y = torch.from_numpy(fx).unsqueeze(0) # Shape: [1, 128, 128, 128]

        return X.float(), Y.float()