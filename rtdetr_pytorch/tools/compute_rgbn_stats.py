# tools/compute_rgbn_stats.py

import numpy as np
import os, glob

npy_files = glob.glob('/path/to/dataset/train/*.npy')
channel_sum = np.zeros(4)
channel_sq_sum = np.zeros(4)
pixel_count = 0

for f in npy_files:
    arr = np.load(f).astype(np.float64)
    if arr.shape[0] == 4:
        arr = arr.transpose(1, 2, 0)   # → (H, W, 4)
    arr = arr / 255.0 if arr.max() > 1.0 else arr

    H, W, C = arr.shape
    channel_sum += arr.reshape(-1, C).sum(axis=0)
    channel_sq_sum += (arr ** 2).reshape(-1, C).sum(axis=0)
    pixel_count += H * W

mean = channel_sum / pixel_count
std  = np.sqrt(channel_sq_sum / pixel_count - mean ** 2)
print(f"Mean: {mean.tolist()}")
print(f"Std:  {std.tolist()}")