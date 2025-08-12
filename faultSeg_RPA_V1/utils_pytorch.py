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

    def __len__(self):
        'Denotes the total number of samples (each sample now yields a batch of 4 augmented images)'
        return len(self.data_IDs)

    def __getitem__(self, index):
        ID = self.data_IDs[index]
        gx = np.load(os.path.join(self.dpath, f"{ID}.npy")).astype(np.single)
        fx = np.load(os.path.join(self.fpath, f"{ID}.npy")).astype(np.single)

        gx = np.reshape(gx, self.dim)
        fx = np.reshape(fx, self.dim)

        # ▶ keep your current orientation (WITH the transpose),
        # which makes axis 0 the "vertical", so we rotate in axes (1,2)
        gx = np.transpose(gx)
        fx = np.transpose(fx)

        augmented_X = []
        augmented_Y = []

        for k in range(4): # 0, 90, 180, 270 degrees rotations
            rot_gx = np.rot90(gx, k=k, axes=(1, 2))
            rot_fx = np.rot90(fx, k=k, axes=(1, 2))

            # Random vertical flip for each rotated version
            if np.random.rand() < 0.5: # 50% chance to flip
                rot_gx = rot_gx[::-1, :, :].copy()
                rot_fx = rot_fx[::-1, :, :].copy()
            
            augmented_X.append(torch.from_numpy(rot_gx).unsqueeze(0))
            augmented_Y.append(torch.from_numpy(rot_fx).unsqueeze(0))

        # Stack the augmented images to form a batch of 4
        X_batch = torch.cat(augmented_X, dim=0)
        Y_batch = torch.cat(augmented_Y, dim=0)

        return X_batch.float(), Y_batch.float()