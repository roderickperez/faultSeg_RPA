import numpy as np
import torch
from torch.utils.data import Dataset
import os

# class DataGenerator(Dataset):
#     'Generates data for PyTorch'
#     def __init__(self, dpath, fpath, data_IDs, dim=(128, 128, 128), n_channels=1):
#         'Initialization'
#         self.dim = dim
#         self.dpath = dpath
#         self.fpath = fpath
#         self.data_IDs = data_IDs
#         self.n_channels = n_channels
#         # The original code's augmentation created a batch of 2 from one sample.
#         # We replicate this by doubling the dataset length and using the index
#         # to decide whether to apply the flip augmentation.
#         self.augmented_length = len(self.data_IDs) * 2

#     def __len__(self):
#         'Denotes the total number of samples'
#         return self.augmented_length

#     def __getitem__(self, index):
#         'Generates one sample of data'
#         # Determine original data ID and if augmentation should be applied
#         original_index = index // 2
#         apply_flip = (index % 2 == 1)
        
#         ID = self.data_IDs[original_index]

#         # Load data
#         gx = np.load(os.path.join(self.dpath, f"{ID}.npy")).astype(np.single)
#         fx = np.load(os.path.join(self.fpath, f"{ID}.npy")).astype(np.single)
        
#         gx = np.reshape(gx, self.dim)
#         fx = np.reshape(fx, self.dim)

#         # Normalize seismic data (matched Keras version by removing epsilon)
#         xm = np.mean(gx)
#         xs = np.std(gx)
#         gx = (gx - xm) / xs

#         # Transpose to (n1, n2, n3)
#         gx = np.transpose(gx)
#         fx = np.transpose(fx)
        
#         # Apply flip augmentation if required
#         if apply_flip:
#             gx = np.flipud(gx).copy() # Use .copy() to avoid negative stride issues
#             fx = np.flipud(fx).copy()

#         # Add channel dimension and convert to PyTorch tensors
#         # PyTorch expects (C, D, H, W)
#         X = torch.from_numpy(gx).unsqueeze(0) # Shape: [1, 128, 128, 128]
#         Y = torch.from_numpy(fx).unsqueeze(0) # Shape: [1, 128, 128, 128]

#         return X.float(), Y.float()
    

class DataGenerator(Dataset):
    def __init__(self, dpath, fpath, data_IDs, dim=(128,128,128),
                 split="train", augment=False, vertical_axis_post=0):
        self.dpath = dpath
        self.fpath = fpath
        self.data_IDs = data_IDs
        self.dim = dim

        # NEW:
        self.split = split              # "train" | "val" | "test"
        self.augment = augment          # enable only for training
        # Because you call np.transpose(...) with no axes (reverses to (2,1,0)),
        # vertical=z ends up at axis 0 after that step → default to 0 here.
        self.vertical_axis_post = vertical_axis_post

    def __len__(self):
        base = len(self.data_IDs)
        # Only double when augmenting in train (original + vertically flipped)
        if self.split == "train" and self.augment:
            return base * 2
        return base

    def __getitem__(self, index):
        # 1) keep your “double-length” pairing logic
        if self.split == "train" and self.augment:
            original_index = index // 2
            apply_flip = (index % 2 == 1)
        else:
            original_index = index
            apply_flip = False

        ID = self.data_IDs[original_index]

        # 2) load + reshape
        gx = np.load(os.path.join(self.dpath, f"{ID}.npy")).astype(np.single)
        fx = np.load(os.path.join(self.fpath, f"{ID}.npy")).astype(np.single)
        gx = np.reshape(gx, self.dim)
        fx = np.reshape(fx, self.dim)

        # 3) per-cube z-score (match your Keras behavior: no epsilon)
        xm = gx.mean(); xs = gx.std()
        gx = (gx - xm) / xs

        # 4) transpose — your current code uses a bare np.transpose()
        #    NOTE: this reverses axes to (2,1,0), so vertical=z becomes axis 0
        gx = np.transpose(gx).copy()
        fx = np.transpose(fx).copy()

        # 5) make labels binary (in case generator wrote 1/2/3)
        fx = (fx > 0).astype(np.single)

        # 6) augmentation: rotate around vertical + deterministic vertical flip
        if self.split == "train" and self.augment:
            vertical_axis = self.vertical_axis_post   # 0 with your current transpose
            horiz_axes = tuple(ax for ax in range(3) if ax != vertical_axis)
            
            # DEBUG (remove later)
            print(f"[DBG] split={self.split} augment={self.augment} "
                  f"vertical_axis_post={vertical_axis} horiz_axes={horiz_axes} k={k} flip={apply_flip}")

            # random rotation k in {0,1,2,3} in the horizontal plane
            k = np.random.randint(0, 4)
            if k:
                gx = np.rot90(gx, k=k, axes=horiz_axes).copy()
                fx = np.rot90(fx, k=k, axes=horiz_axes).copy()

            # flipped copy is the “paired” sample
            if apply_flip:
                gx = np.flip(gx, axis=vertical_axis).copy()
                fx = np.flip(fx, axis=vertical_axis).copy()

        # 7) to tensors (C, D, H, W)
        X = torch.from_numpy(np.ascontiguousarray(gx)).unsqueeze(0).float()
        Y = torch.from_numpy(np.ascontiguousarray(fx)).unsqueeze(0).float()
        return X, Y