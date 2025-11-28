import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from typing import Optional, Callable, Tuple
import pathlib


class SlowFastKineticsDataset(Dataset):
    """
    Creats Kinetics-400 Dataset object for SlowFast model
    """
    def __init__(self, slow_dataset, fast_dataset):
        self.slow_dataset =slow_dataset
        self.fast_dataset = fast_dataset
        self.min_length = min(len(slow_dataset), len(fast_dataset))
        

    def __len__(self):
        return self.min_length

    def __getitem__(self, idx):
        #get same frame from both datsets
        slow_video, slow_audio, slow_label = self.slow_dataset[idx]
        fast_video, fast_audio, fast_label = self.fast_dataset[idx]

        #check that labels match
        if slow_label == fast_label:
            print('labels for slow and fast are the same')
            return slow_video, fast_video, slow_label
        else:
            print('labels not the same, will use slow label')
            return slow_video, fast_video, slow_label


def kinnetics_collate_fn(batch):
    """
    Kinetics returns (video, audio, label) tuples
    We only need video and label for our training
    Convert from [T, C, H, W] to [B, T, C, H, W] for our model
    """
    videos = []
    labels = []
    
    for video, audio, label in batch:
        videos.append(video)
        labels.append(label)
    
    #stack videos and adjust dimensions
    #Input: list of [T, C, H, W] -> Output: [B, T, C, H, W]
    videos = torch.stack(videos)
    videos = videos.permute(0, 1, 2, 3, 4)
    labels = torch.tensor(labels)

    return videos, labels

def slowfast_collate_fn(batch):
    """
    Collate function for SlowFast dataset that returns slow and fast pathways
    """
    
    slow_videos = []
    fast_videos = []
    labels = []

    for slow_video, fast_video, label in batch:
        slow_videos.append(slow_video)
        fast_videos.append(fast_video)
        labels.append(label)

    #stack and keep dimensions
    slow_videos = torch.stack(slow_videos)
    slow_videos = slow_videos.permute(0, 1, 2, 3, 4)
    fast_videos = torch.stack(fast_videos)
    fast_videos = fast_videos.permute(0, 1, 2, 3, 4)
    labels = torch.tensor(labels)

    return (slow_videos, fast_videos), labels


def get_default_train_transform(resolution: int = 224) -> transforms.Compose:
    """
    Get default training trasforms
    """
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.RandomResizedCrop(resolution, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], 
                           std=[0.22803, 0.22145, 0.216989])
    ])


def get_default_val_transform(resolution: int = 224) -> transforms.Compose:
    """
    Get default validating transforms
    """
    return transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37654], std=[0.22803, 0.22145, 0.216989])
        ])

def get_num_classes(num_classes_str: str = '400') -> int:
    return int(num_classes_str)

def create_kinetics_datasets(root: str,
                                split: str = 'train',
                                frames_per_clip: int = 32,
                                num_classes: str = '400', 
                                frame_rate: Optional[float] = None,
                                step_between_clips: int = 1,
                                num_workers: int = 4,
                                transform: Optional[Callable] = None,
                                download: bool = True
                                ) -> datasets.Kinetics:
    
    """
    Create Kinetics-400 dataloaders using Pytorch's official dataset

    Args:
        root: Rott directory where kinetics dataset is stored
        frames_per_clip: Numer of frames in each cliup
        num_classes: '400', '600', or '700' depending on version of Kinetics dataset (but I only use '400')
        frame_rate: If None, uses original video frame rate
        step_between_clips: step between each selected frame
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory or not
        resolution: Spatial resolution in frames
        train_transform: Custom transforms for training
        val_transforms: Custom transforms for validation
        download: Whether to download the dataset

    Returns:
        train_loader: Train DataLoader
        val_loader: Valudation DataLoader
        num_classes_int: Number of classes as integer
    """

    #default transforms
    if transform is None:
        if split == 'train':
            transform = get_default_train_transform()
        else:
            transform = get_default_val_transform()

    dataset = datasets.Kinetics(
        root=root,
        frames_per_clip=frames_per_clip,
        num_classes=num_classes,
        split=split,
        frame_rate=frame_rate,
        step_between_clips=step_between_clips,
        transform=transform,
        download=download,
        num_workers=num_workers
    )

    return dataset

def create_kinetics_dataloaders(root: str,
                                         frames_per_clip: int = 32,
                                         num_classes: str = '400',
                                         frame_rate: Optional[int] = None,
                                         step_between_clips: int = 1,
                                         batch_size: int = 8,
                                         num_workers: int = 4,
                                         pin_memory: bool = True,
                                         resolution: int = 224,
                                         train_transform: Optional[Callable] = None,
                                         val_transform: Optional[Callable] = None,
                                         download: bool = True
                                         ) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create Kinetics-400 dataloaders using PyTorch's official dataset
    
    Args:
        root: Root directory where kinetics dataset is stored/will be downloaded
        frames_per_clip: Number of frames in each clip
        num_classes: '400', '600', or '700' for different Kinetics versions
        frame_rate: If None, uses original video frame rate
        step_between_clips: Step between consecutive clips
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        resolution: Spatial resolution of frames
        train_transform: Custom transforms for training
        val_transform: Custom transforms for validation
        download: Whether to download the dataset
    
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader  
        num_classes_int: Number of classes as integer
    """

    #create dataloaders
    train_dataset = create_kinetics_datasets(root=root,
        frames_per_clip=frames_per_clip,
        num_classes=num_classes,
        split='train',
        frame_rate=frame_rate,
        step_between_clips=step_between_clips,
        transform=train_transform,
        download=download,
        num_workers=num_workers
        )
    
    val_dataset = create_kinetics_datasets(root=root,
        frames_per_clip=frames_per_clip,
        num_classes=num_classes,
        split='val',
        frame_rate=frame_rate,
        step_between_clips=step_between_clips,
        transform=train_transform,
        download=download,
        num_workers=num_workers
        )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    num_classes_int = get_num_classes(num_classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=kinnetics_collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=kinnetics_collate_fn,
        drop_last=False
    )

    return train_loader, val_loader, num_classes_int

def create_slowfast_kinetics_dataloaders(root: str,
                                         slow_frames: int = 8,
                                         fast_frames: int = 32,
                                         num_classes: str = '400',
                                         frame_rate: Optional[int] = None,
                                         batch_size: int = 8,
                                         num_workers: int = 4,
                                         resolution: int = 224,
                                         download: bool = True
                                         ) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create dataloaders specifically optimized for SlowFast architecture
    
    Args:
        slow_frames: Number of frames for slow pathway
        fast_frames: Number of frames for fast pathway
    """

    #create datasets with different frame lengths for slow and fast pathway
    train_dataset_slow = create_kinetics_datasets(root=root,
        split='train',
        frames_per_clip=slow_frames,
        num_classes=num_classes,
        frame_rate=frame_rate,
        transform=get_default_train_transform(resolution),
        download=download,
        num_workers=num_workers
        )

    train_dataset_fast = create_kinetics_datasets(root=root,
        split='train',
        frames_per_clip=fast_frames,
        num_classes=num_classes,
        frame_rate=frame_rate,
        transform=get_default_train_transform(resolution),
        download=download,
        num_workers=num_workers)

    train_dataset = SlowFastKineticsDataset(train_dataset_slow, train_dataset_fast)

    val_dataset_slow = create_kinetics_datasets(root=root,
        split='val',
        frames_per_clip=slow_frames,
        num_classes=num_classes,
        frame_rate=frame_rate,
        transform=get_default_val_transform(resolution),
        download=download,
        num_workers=num_workers)

    val_dataset_fast = create_kinetics_datasets(root=root,
        split='val',
        frames_per_clip=fast_frames,
        num_classes=num_classes,
        frame_rate=frame_rate,
        transform=get_default_val_transform(resolution),
        download=download,
        num_workers=num_workers)
    
    val_dataset = SlowFastKineticsDataset(val_dataset_slow, val_dataset_fast)

    train_loader = DataLoader(train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=slowfast_collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=slowfast_collate_fn,
        drop_last=False
    )

    num_classes_int = get_num_classes(num_classes)

    return train_loader, val_loader, num_classes_int

def test_kinetics_loader():
    """
    Test kinetics Dataloader
    """
    train_loader, val_loader, num_classes = create_kinetics_dataloaders(
        root="./data/kinetics",
        frames_per_clip=16,  # Smaller for testing
        batch_size=2,
        num_workers=0,
        download=True
    )
    
    print(f"Number of classes: {num_classes}")
    
    # Test one batch
    for videos, labels in train_loader:
        print(f"Videos shape: {videos.shape}")  # Should be [2, 16, 3, 224, 224]
        print(f"Labels shape: {labels.shape}")  # Should be [2]
        print(f"Sample labels: {labels}")
        break
    
    return train_loader, val_loader, num_classes
    
            
if __name__ == "__main__":
    test_kinetics_loader()
    
