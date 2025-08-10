import pathlib
import numpy as np
import pandas as pd
import torch
import torchvision
from omegaconf import DictConfig
from typing import Optional, Tuple, Dict, Any

from PIL import Image
import av


class ClipsDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset class for loading and processing video clips for action recognition.
    
    This dataset handles surgical video clips from the SLAM dataset, processing them into
    fixed-length temporal sequences with spatial transformations for training deep learning models.
    
    Args:
        config (DictConfig): Hydra configuration containing dataset parameters
        split (str): Dataset split ('train', 'val', or 'test')
        transform (Optional[torchvision.transforms.Compose]): Custom transforms to apply
        
    Attributes:
        config: Configuration object containing all dataset parameters
        data_path: Path to the dataset root directory
        data_label: DataFrame containing video paths and labels
        resize_shape: Tuple for resizing frames (width, height)
        time_size: Number of frames to extract per video clip
        label_dict: Mapping from action names to integer labels
        transform: Image transformation pipeline
    """
    
    def __init__(
        self, 
        config: DictConfig, 
        split: str = 'train',
        transform: Optional[torchvision.transforms.Compose] = None
    ):
        """
        Initialize the ClipsDataset with Hydra configuration.
        
        Args:
            config: Hydra configuration object
            split: Which data split to use ('train', 'val', 'test')
            transform: Optional custom transformation pipeline
        """
        self.config = config
        self.split = split
        
        # Extract configuration parameters
        self.data_path = pathlib.Path(config.dataset.data_path)
        self.resize_shape = tuple(config.video_processing.resize_shape)
        self.time_size = config.video_processing.time_size
        self.label_dict = dict(config.dataset.classes)
        
        # Load the appropriate CSV file based on split
        csv_filename = getattr(config.dataset, f"{split}_csv")
        csv_path = self.data_path / csv_filename
        self.data_label = pd.read_csv(csv_path)
        
        # Set up transformations
        if transform is None:
            self.transform = self._create_default_transform(split)
        else:
            self.transform = transform
    
    def _create_default_transform(self, split: str) -> torchvision.transforms.Compose:
        """
        Create default transformation pipeline based on split type and config.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            
        Returns:
            Composed transformation pipeline
        """
        transforms_list = []
        
        # Add augmentations for training split
        if split == 'train' and hasattr(self.config.augmentation, 'train'):
            aug_config = self.config.augmentation.train
            
            if aug_config.get('random_horizontal_flip', False):
                transforms_list.append(torchvision.transforms.RandomHorizontalFlip())
            
            if 'random_rotation' in aug_config:
                transforms_list.append(
                    torchvision.transforms.RandomRotation(aug_config.random_rotation)
                )
            
            if 'color_jitter' in aug_config:
                cj = aug_config.color_jitter
                transforms_list.append(
                    torchvision.transforms.ColorJitter(
                        brightness=cj.get('brightness', 0),
                        contrast=cj.get('contrast', 0),
                        saturation=cj.get('saturation', 0)
                    )
                )
            
            if 'random_affine' in aug_config:
                ra = aug_config.random_affine
                transforms_list.append(
                    torchvision.transforms.RandomAffine(
                        degrees=ra.get('degrees', 0),
                        translate=tuple(ra.get('translate', [0, 0])),
                        scale=tuple(ra.get('scale', [1.0, 1.0]))
                    )
                )
        
        # Always add tensor conversion as the final transform
        transforms_list.append(torchvision.transforms.ToTensor())
        
        return torchvision.transforms.Compose(transforms_list)
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.data_label)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieve a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple containing:
                - frames: Tensor of shape (time_size, channels, height, width)
                - label: Integer label for the action class
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Extract video path and label from CSV
        relative_video_path = self.data_label.iloc[idx, 1]
        video_name = pathlib.Path(relative_video_path).name
        video_path = self.data_path / self.config.dataset.video_subdir / video_name
        action_name = self.data_label.iloc[idx, 2]
        label = self.label_dict[action_name]
        
        # Load and process video frames
        frames = self._load_video_frames(video_path)
        
        # Normalize temporal dimension
        frames = self._normalize_temporal_dimension(frames)
        
        # Stack frames into a single tensor
        frames = torch.stack(frames)
        
        return frames, label
    
    def _load_video_frames(self, video_path: pathlib.Path) -> list:
        """
        Load and process frames from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of processed frame tensors
        """
        container = av.open(str(video_path))
        frames = []
        
        # Decode all frames from the video
        for frame in container.decode(video=0):
            # Convert frame to RGB numpy array
            img = frame.to_ndarray(format='rgb24')
            
            # Resize to target dimensions
            img = Image.fromarray(img).resize(self.resize_shape)
            
            # Apply transformations if available
            if self.transform:
                img = self.transform(img)
            
            frames.append(img)
        
        container.close()
        return frames
    
    def _normalize_temporal_dimension(self, frames: list) -> list:
        """
        Normalize the temporal dimension to the target time_size.
        
        Handles videos that are shorter or longer than the target by padding
        with the last frame or truncating respectively.
        
        Args:
            frames: List of frame tensors
            
        Returns:
            List of frames normalized to time_size length
        """
        if len(frames) < self.time_size:
            # Pad with the last frame if video is too short
            frames += [frames[-1]] * (self.time_size - len(frames))
        elif len(frames) > self.time_size:
            # Truncate if video is too long
            frames = frames[:self.time_size]
        
        return frames
    
    def get_class_names(self) -> Dict[int, str]:
        """
        Get mapping from class indices to class names.
        
        Returns:
            Dictionary mapping integer labels to action names
        """
        return {v: k for k, v in self.label_dict.items()}
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset.
        
        Returns:
            Dictionary containing dataset statistics and configuration
        """
        return {
            'split': self.split,
            'total_samples': len(self),
            'num_classes': len(self.label_dict),
            'class_names': list(self.label_dict.keys()),
            'temporal_size': self.time_size,
            'spatial_size': self.resize_shape,
            'data_path': str(self.data_path),
            'config': self.config
        }

def get_input_size_from_dataset(data_loader):
    for x, _ in data_loader:
        return x.shape