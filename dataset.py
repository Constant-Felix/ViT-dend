import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Try to import spikingjelly. If not installed, print an error.
try:
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
    from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
except ImportError:
    print("SpikingJelly not installed. Please install it with: pip install spikingjelly")
    exit()

import os

def load_all_datasets(data_root='./data', batch_size=64):
    """
    Loads CIFAR10, CIFAR100, DVSGesture, and CIFAR10-DVS datasets.
    
    Args:
        data_root (str): The root directory where all datasets will be stored.
                         Based on your image, this is './data/'.
        batch_size (int): Batch size for the DataLoaders.
    """
    
    # Ensure the root data directory exists
    os.makedirs(data_root, exist_ok=True)
    
    print(f"Loading datasets into '{data_root}'...")
    
    # --- 1. PyTorch: CIFAR10 ---
    # PyTorch will automatically create and use the 'cifar-10-batches-py' subfolder
    # inside the specified root.
    print("\nLoading CIFAR10...")
    cifar10_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        cifar10_trainset = torchvision.datasets.CIFAR10(
            root=data_root, 
            train=True,
            download=True, 
            transform=cifar10_transform
        )
        cifar10_train_loader = DataLoader(
            cifar10_trainset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=2
        )
        
        cifar10_testset = torchvision.datasets.CIFAR10(
            root=data_root, 
            train=False,
            download=True, 
            transform=cifar10_transform
        )
        cifar10_test_loader = DataLoader(
            cifar10_testset, 
            batch_size=batch_size,
            shuffle=False, 
            num_workers=2
        )
        
        print(f"CIFAR10 train set size: {len(cifar10_trainset)}")
        print(f"CIFAR10 test set size: {len(cifar10_testset)}")
        print("CIFAR10 loaded successfully.")

    except Exception as e:
        print(f"Error loading CIFAR10: {e}")

    # --- 2. PyTorch: CIFAR100 ---
    # PyTorch will automatically create and use the 'cifar-100-python' subfolder.
    print("\nLoading CIFAR100...")
    cifar100_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        cifar100_trainset = torchvision.datasets.CIFAR100(
            root=data_root, 
            train=True,
            download=True, 
            transform=cifar100_transform
        )
        cifar100_train_loader = DataLoader(
            cifar100_trainset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=2
        )
        
        cifar100_testset = torchvision.datasets.CIFAR100(
            root=data_root, 
            train=False,
            download=True, 
            transform=cifar100_transform
        )
        cifar100_test_loader = DataLoader(
            cifar100_testset, 
            batch_size=batch_size,
            shuffle=False, 
            num_workers=2
        )
        
        print(f"CIFAR100 train set size: {len(cifar100_trainset)}")
        print(f"CIFAR100 test set size: {len(cifar100_testset)}")
        print("CIFAR100 loaded successfully.")
        
    except Exception as e:
        print(f"Error loading CIFAR100: {e}")

    # --- 3. SpikingJelly: DVSGesture ---
    # We specify the exact sub-directory 'DVSGesturedataset' as the root.
    # The parameters are set to match your directory structure, e.g.,
    # 'frames_number_10_split_by_number'.
    print("\nLoading DVSGesture...")
    gesture_root = os.path.join(data_root, 'DVSGesturedataset')
    
    # Parameters for frame-based processing, as implied by your folder structure
    gesture_frames_number = 10
    gesture_split_by = 'number'
    
    try:
        # SpikingJelly datasets need to be downloaded and processed.
        # This might take some time on the first run.
        gesture_trainset = DVS128Gesture(
            root=gesture_root,
            train=True,
            data_type='frame',
            frames_number=gesture_frames_number,
            split_by=gesture_split_by
        )
        gesture_train_loader = DataLoader(
            gesture_trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        gesture_testset = DVS128Gesture(
            root=gesture_root,
            train=False,
            data_type='frame',
            frames_number=gesture_frames_number,
            split_by=gesture_split_by
        )
        gesture_test_loader = DataLoader(
            gesture_testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        print(f"DVSGesture train set size: {len(gesture_trainset)}")
        print(f"DVSGesture test set size: {len(gesture_testset)}")
        print("DVSGesture loaded successfully.")
        
    except Exception as e:
        print(f"Error loading DVSGesture: {e}")
        print("Note: SpikingJelly datasets often require manual download first.")
        print("Please check the SpikingJelly documentation for setup instructions if download fails.")


    # --- 4. SpikingJelly: CIFAR10-DVS ---
    # We specify the 'cifar10-dvs' sub-directory as the root.
    # The library will handle creating processed versions like 'cifar10-dvs-tet'.
    print("\nLoading CIFAR10-DVS...")
    cifar_dvs_root = os.path.join(data_root, 'cifar10-dvs')
    
    # Parameters for frame-based processing
    cifar_dvs_frames_number = 10
    cifar_dvs_split_by = 'number'

    try:
        cifar_dvs_trainset = CIFAR10DVS(
            root=cifar_dvs_root,
            train=True,
            data_type='frame',
            frames_number=cifar_dvs_frames_number,
            split_by=cifar_dvs_split_by
        )
        cifar_dvs_train_loader = DataLoader(
            cifar_dvs_trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        cifar_dvs_testset = CIFAR10DVS(
            root=cifar_dvs_root,
            train=False,
            data_type='frame',
            frames_number=cifar_dvs_frames_number,
            split_by=cifar_dvs_split_by
        )
        cifar_dvs_test_loader = DataLoader(
            cifar_dvs_testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        print(f"CIFAR10-DVS train set size: {len(cifar_dvs_trainset)}")
        print(f"CIFAR10-DVS test set size: {len(cifar_dvs_testset)}")
        print("CIFAR10-DVS loaded successfully.")
        
    except Exception as e:
        print(f"Error loading CIFAR10-DVS: {e}")
        print("Note: SpikingJelly datasets often require manual download first.")
        print("Please check the SpikingJelly documentation for setup instructions if download fails.")

    print("\n--- All dataset loading initiated. ---")
    
    # You can return the loaders if you want to use them elsewhere
    return {
        'cifar10_train': cifar10_train_loader,
        'cifar10_test': cifar10_test_loader,
        'cifar100_train': cifar100_train_loader,
        'cifar100_test': cifar100_test_loader,
        'gesture_train': gesture_train_loader,
        'gesture_test': gesture_test_loader,
        'cifar_dvs_train': cifar_dvs_train_loader,
        'cifar_dvs_test': cifar_dvs_test_loader
    }

if __name__ == '__main__':
    # Set the main data directory
    DATA_DIR = './data'
    
    # Load all datasets
    loaders = load_all_datasets(data_root=DATA_DIR, batch_size=32)
    
    # Example: Iterate over one batch from each training loader
    print("\nTesting DataLoaders by fetching one batch from each train set...")
    
    try:
        cifar10_data, cifar10_labels = next(iter(loaders['cifar10_train']))
        print(f"CIFAR10 batch shape: {cifar10_data.shape}, Labels shape: {cifar10_labels.shape}")
    except Exception as e:
        print(f"Could not fetch CIFAR10 batch: {e}")
        
    try:
        cifar100_data, cifar100_labels = next(iter(loaders['cifar100_train']))
        print(f"CIFAR100 batch shape: {cifar100_data.shape}, Labels shape: {cifar100_labels.shape}")
    except Exception as e:
        print(f"Could not fetch CIFAR100 batch: {e}")

    try:
        # SpikingJelly frame data shape is usually [B, T, C, H, W]
        gesture_data, gesture_labels = next(iter(loaders['gesture_train']))
        print(f"DVSGesture batch shape: {gesture_data.shape}, Labels shape: {gesture_labels.shape}")
    except Exception as e:
        print(f"Could not fetch DVSGesture batch: {e}")

    try:
        # SpikingJelly frame data shape is usually [B, T, C, H, W]
        cifar_dvs_data, cifar_dvs_labels = next(iter(loaders['cifar_dvs_train']))
        print(f"CIFAR10-DVS batch shape: {cifar_dvs_data.shape}, Labels shape: {cifar_dvs_labels.shape}")
    except Exception as e:
        print(f"Could not fetch CIFAR10-DVS batch: {e}")
