import torch
import torchvision
import torchvision.transforms as transforms


# Download and load the training set
trainset = torchvision.datasets.CIFAR10(
    root='./data/cifar10',      # Directory to download/store the dataset (change as needed)
    train=True,         # True for training split (50,000 images)
    download=True     # Set to True to download if not present
)

# Download and load the test set
testset = torchvision.datasets.CIFAR10(
    root='./data/cifar10',
    train=False,        # False for test split (10,000 images)
    download=True

)