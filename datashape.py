import torch
import numpy as np

data = np.load("data/cifar10-dvs/cifar10-dvs/frames_number_16_split_by_number/airplane/cifar10_airplane_1.npz")

a = data['frames']
print(a.shape)

for i in range(12):
    print((a[8]==i).sum().item())
