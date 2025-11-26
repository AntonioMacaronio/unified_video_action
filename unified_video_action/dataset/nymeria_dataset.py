"""
Nymeria dataset wrapper for UVA (Unified Video Action) model.

This module wraps the NymeriaDataset from the pip-installed nymeria package
to provide data in the format expected by UVA for video generation training.
"""

from typing import Dict
import torch
import numpy as np
import copy
from pathlib import Path
import os
import pickle

# Import from the pip-installed nymeria package
from nymeria import NymeriaDataset, NymeriaTrainingSeq, BatchedNymeriaTrainingSeq

from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.model.common.normalizer import LinearNormalizer
from unified_video_action.dataset.base_dataset import BaseImageDataset
from unified_video_action.common.normalize_util import get_image_range_normalizer
import torchvision.transforms as transforms
import torchvision


class NymeriaUVADataset(BaseImageDataset):
    """
    Wrapper for Nymeria data to work with UVA model.

    This dataset loads 150-frame sequences from the nymeria HDF5 files and
    prepares them for video generation training (no action prediction).
    """

    def __init__(
        self,
        data_dir,
        image_resolution=224,
        sequence_length=150,
        val_ratio=0.02,
        seed=42,
        data_aug=False,
        file_pattern="*.h5",
        language_emb_model=None,
        normalizer_type=None,
    ):
        """
        Args:
            data_dir: Path to directory containing Nymeria HDF5 files
            image_resolution: Target resolution for images (default: 96)
            sequence_length: Number of frames per sequence (default: 150)
            val_ratio: Ratio of data to use for validation (default: 0.02)
            seed: Random seed for train/val split
            data_aug: Whether to apply data augmentation
            file_pattern: Glob pattern for HDF5 files (default: "*.h5")
            language_emb_model: Language embedding model (not used for video-only training)
            normalizer_type: Action normalizer type (not used for video-only training)
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.image_resolution = image_resolution
        self.sequence_length = sequence_length
        self.val_ratio = val_ratio
        self.seed = seed
        self.data_aug = data_aug

        # Load the nymeria dataset
        self.nymeria_dataset = NymeriaDataset(data_dir, file_pattern=file_pattern)

        # Create train/val split
        np.random.seed(seed)
        n_episodes = len(self.nymeria_dataset)
        indices = np.arange(n_episodes)
        np.random.shuffle(indices)

        n_val = int(n_episodes * val_ratio)
        self.val_indices = indices[:n_val]
        self.train_indices = indices[n_val:]

        # Start with train indices
        self.is_train = True
        self.active_indices = self.train_indices # NOTE: active_indices is the indices of the sequences that are currently being used. =

        # Image transforms
        self.resize_transform = transforms.Resize(
            (image_resolution, image_resolution),
            antialias=True
        )

        print(f"NymeriaUVADataset initialized:")
        print(f"  Total episodes: {n_episodes}")
        print(f"  Train episodes: {len(self.train_indices)}")
        print(f"  Val episodes: {len(self.val_indices)}")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Image resolution: {image_resolution}x{image_resolution}")

    def get_validation_dataset(self):
        """Return a copy of this dataset configured for validation."""
        val_set = copy.copy(self)
        val_set.is_train = False
        val_set.active_indices = self.val_indices
        val_set.data_aug = False  # Disable augmentation for validation
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        """
        Return normalizer for the dataset.
        For video-only training, we only need image normalization.
        """
        normalizer = LinearNormalizer()
        # For video generation, we create dummy actions with zero mean/std
        # The actual normalization will be handled by the image normalizer
        dummy_action = np.zeros((100, 1))  # (episodes, action_dim)
        normalizer.fit(data={"action": dummy_action}, last_n_dims=1, mode=mode, **kwargs)
        normalizer["image"] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        """Return all actions. For video-only training, returns dummy actions."""
        # Return dummy actions with shape (n_samples, action_dim)
        return torch.zeros(len(self), 1)

    def __len__(self) -> int:
        return len(self.active_indices)

    def _process_image(self, image_np):
        """Process a single image: resize and optionally augment.
        Args:
            image_np: numpy array of shape (3, 1408, 1408) with values in [0, 255]
        Returns:
            torch tensor of shape (3, image_resolution, image_resolution) in [0, 1]
        """
        # Convert to torch tensor and normalize to [0, 1] and resize to target resolution
        image_tensor = torch.from_numpy(image_np) / np.float32(255)
        image_tensor = self.resize_transform(image_tensor)
        return image_tensor

    def _apply_video_augmentation(self, video_tensor):
        """Apply consistent augmentation across all frames in the video.
        Args:
            video_tensor: torch tensor of shape (T, 3, H, W)
        Returns:
            Augmented video tensor of same shape
        """
        # Use same random seed for all frames to ensure consistent augmentation
        video_seed = torch.randint(0, 10000, (1,)).item()

        def consistent_augmentations(frame):
            torch.manual_seed(video_seed)

            frame_size = self.image_resolution
            augmentation = transforms.Compose([
                torchvision.transforms.RandomApply(
                    [torchvision.transforms.RandomCrop(size=int(frame_size * 0.95))],
                    p=0.5,
                ),
                torchvision.transforms.Resize(size=frame_size, antialias=True),
                torchvision.transforms.RandomApply(
                    [torchvision.transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))],
                    p=0.5,
                ),
            ])
            return augmentation(frame)

        augmented_frames = torch.stack([consistent_augmentations(frame) for frame in video_tensor])
        return augmented_frames

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example.
        Returns:
            dict with:
                obs: dict with:
                    image: torch.Tensor (T, 3, H, W) normalized to [0, 1]
                action: torch.Tensor (T, 1) dummy actions (all zeros)
        """
        # Map idx to actual dataset index
        dataset_idx = self.active_indices[idx]

        # Load the sequence from nymeria dataset and pad or trim to desired sequence length
        seq: NymeriaTrainingSeq = self.nymeria_dataset[dataset_idx]
        seq, _ = seq.pad_or_trim_sequence(self.sequence_length)

        # Process all frames
        # egoview_RGB shape: (T, 3, 1408, 1408) uint8 [0, 255]
        processed_frames = []
        for t in range(self.sequence_length):
            frame = self._process_image(seq.egoview_RGB[t])  # (3, H, W)
            processed_frames.append(frame)
        video_tensor = torch.stack(processed_frames)  # (T, 3, H, W)

        # Apply augmentation if enabled
        if self.data_aug:
            video_tensor = self._apply_video_augmentation(video_tensor)

        # Create dummy actions (zeros) for video-only training
        # Shape: (T, 1)
        dummy_actions = torch.zeros(self.sequence_length, 1, dtype=torch.float32)

        data = {
            "obs": {
                "image": video_tensor,  # (T, 3, H, W) in [0, 1]
            },
            "action": dummy_actions,  # (T, 1)
        }

        return data


if __name__ == "__main__":
    # Test the dataset
    print("=" * 80)
    print("Testing NymeriaUVADataset")
    print("=" * 80)

    data_dir = Path("/nfs/antzhan/nymeria/hdf5")  # Update this path

    if data_dir.exists():
        # Create dataset
        dataset = NymeriaUVADataset(
            data_dir=data_dir,
            image_resolution=96,
            sequence_length=150,
            val_ratio=0.02,
            data_aug=True,
        )

        print(f"\nDataset length: {len(dataset)}")

        # Get validation dataset
        val_dataset = dataset.get_validation_dataset()
        print(f"Validation dataset length: {len(val_dataset)}")

        # Test loading a sample
        print("\nLoading sample...")
        sample = dataset[0]

        print(f"\nSample structure:")
        print(f"  obs['image'] shape: {sample['obs']['image'].shape}")
        print(f"  obs['image'] dtype: {sample['obs']['image'].dtype}")
        print(f"  obs['image'] min/max: {sample['obs']['image'].min():.3f} / {sample['obs']['image'].max():.3f}")
        print(f"  action shape: {sample['action'].shape}")
        print(f"  action dtype: {sample['action'].dtype}")

        # Test normalizer
        print("\nTesting normalizer...")
        normalizer = dataset.get_normalizer()
        print(f"Normalizer keys: {list(normalizer.keys())}")

    else:
        print(f"Data directory not found: {data_dir}")
        print("Please update the data_dir variable in the __main__ section")
