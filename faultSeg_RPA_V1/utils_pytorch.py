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
        self.augmented_length = len(self.data_IDs) * 8

    def __len__(self):
        'Denotes the total number of samples'
        return self.augmented_length

    def __getitem__(self, index):
        original_index = index // 8
        rot_k          = (index % 8) % 4     # 0,1,2,3
        apply_flip     = (index % 8) // 4    # 0 or 1

        ID = self.data_IDs[original_index]
        gx = np.load(os.path.join(self.dpath, f"{ID}.npy")).astype(np.single)
        fx = np.load(os.path.join(self.fpath, f"{ID}.npy")).astype(np.single)

        gx = np.reshape(gx, self.dim)
        fx = np.reshape(fx, self.dim)

        # z-score
        xm, xs = np.mean(gx), np.std(gx)
        gx = (gx - xm) / xs

        # ▶ keep your current orientation (WITH the transpose),
        # which makes axis 0 the "vertical", so we rotate in axes (1,2)
        gx = np.transpose(gx)
        fx = np.transpose(fx)

        # 90° rotations around the vertical axis
        gx = np.rot90(gx, k=rot_k, axes=(1, 2))
        fx = np.rot90(fx, k=rot_k, axes=(1, 2))

        # vertical flip
        if apply_flip:
            gx = gx[::-1, :, :].copy()
            fx = fx[::-1, :, :].copy()

        X = torch.from_numpy(gx).unsqueeze(0)
        Y = torch.from_numpy(fx).unsqueeze(0)
        return X.float(), Y.float()