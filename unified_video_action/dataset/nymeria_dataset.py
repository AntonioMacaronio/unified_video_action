"""
Nymeria dataset wrapper for UVA (Unified Video Action) model.

This module wraps the NymeriaDataset from the pip-installed nymeria package
to provide data in the format expected by UVA for video generation training.
"""

from typing import Dict
import torch
from torch.utils.data import Subset
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

from transforms import SE3, SO3

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
        num_datapoints=-1,
        text_embeddings_path=None,
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
            language_emb_model: Language embedding model (e.g., "clip" for CLIP embeddings)
            normalizer_type: Action normalizer type (e.g., "limits" for min-max normalization)
            num_datapoints: Number of datapoints to use (default: -1 for all)
                - Use 1 datapoint to overfit the model to 1 datapoint for debugging purposes
                - This is set in `nymeria.yaml` config file under `dataset.num_datapoints`
            text_embeddings_path: Path to precomputed text embeddings pickle file.
                Generate with: python scripts/precompute_nymeria_text_embeddings.py
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.image_resolution = image_resolution
        self.sequence_length = sequence_length
        self.val_ratio = val_ratio
        self.seed = seed
        self.data_aug = data_aug
        self.num_datapoints = num_datapoints
        self.language_emb_model = language_emb_model

        # Load precomputed text embeddings if provided
        self.text_embeddings = None # this is a dictionary of {filename: (512,) tensor}
        if text_embeddings_path is not None:
            text_embeddings_path = Path(text_embeddings_path)
            if text_embeddings_path.exists():
                print(f"Loading precomputed text embeddings from {text_embeddings_path}")
                with open(text_embeddings_path, "rb") as f:
                    text_emb_data = pickle.load(f)
                    # this is a dictionary with the following keys: 'embeddings', 'texts', 'model', 'embedding_dim'
                    # {
                    #     'embeddings': {filename: (512,) tensor},
                    #     'texts': {filename: text},
                    #     'model': 'openai/clip-vit-base-patch32',
                    #     'embedding_dim': 512
                    # }
                self.text_embeddings = text_emb_data["embeddings"]  # {filename: (512,) tensor}
                print(f"  Loaded {len(self.text_embeddings)} text embeddings (dim={text_emb_data['embedding_dim']})")
            else:
                print(f"WARNING: Text embeddings file not found: {text_embeddings_path}")
                print("  Run: python scripts/precompute_nymeria_text_embeddings.py --data-dir <data_dir>")

        # Load the nymeria dataset with image_resolution for decode-time resize (4-5x faster)
        self.nymeria_dataset = NymeriaDataset(data_dir, file_pattern=file_pattern, image_resolution=image_resolution)
        if self.num_datapoints != -1:
            num_repeats = len(self.nymeria_dataset) // self.num_datapoints
            self.nymeria_dataset = Subset(self.nymeria_dataset, list(range(self.num_datapoints)) * num_repeats)
        
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
        Action dimension is 144 = 6 (CPF change) + 23*6 (joint twists).

        This iterates through the dataset to collect all actions and fits
        the normalizer on the actual data distribution. This is slow but
        only runs once at training start.
        """
        normalizer = LinearNormalizer()

        # Collect all actions from the dataset
        all_actions = self.get_all_actions()  # (N, 144)

        normalizer.fit(data={"action": all_actions.numpy()}, last_n_dims=1, mode=mode, **kwargs)
        normalizer["image"] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        """
        Return all actions from the training set.

        Iterates through all training samples to collect actions.
        Returns tensor of shape (N, 144) where N = len(dataset).
        """
        action_dim = 6 + 23 * 6  # 144
        all_actions = []

        print(f"Collecting all actions from {len(self)} samples for normalizer fitting...")
        for idx in range(len(self)):
            sample = self[idx]
            # sample['action'] has shape (T, 144), we flatten to (T*144) or keep per-timestep
            # For normalizer, we want all action values, so concatenate all timesteps
            actions = sample['action']  # (T, 144)
            all_actions.append(actions)

        # Stack all actions: (N, T, 144) -> reshape to (N*T, 144)
        all_actions = torch.stack(all_actions, dim=0)  # (N, T, 144)
        all_actions = all_actions.reshape(-1, action_dim)  # (N*T, 144)

        print(f"Collected {all_actions.shape[0]} action samples with dimension {action_dim}")
        return all_actions

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
                action: torch.Tensor (T, 144) containing:
                    - CPF change twist (6D): relative SE3 twist from frame i to i+1
                      Format: (vx, vy, vz, omega_x, omega_y, omega_z)
                      First timestep is zeros (no previous frame).
                    - Joint twists (23 joints * 6D = 138D): SE3 twist of each joint
                      relative to CPF at each timestep.
                      T_{cpf, joint} = T_{world, cpf}^-1 @ T_{world, joint}
        """
        # Map idx to actual dataset index
        dataset_idx = self.active_indices[idx]

        # Load the sequence from nymeria dataset and pad or trim to desired sequence length
        seq: NymeriaTrainingSeq = self.nymeria_dataset[dataset_idx]
        seq, _ = seq.pad_or_trim_sequence(self.sequence_length)

        # Convert frames to tensor and normalize to [0, 1]
        # egoview_RGB shape: (T, 3, H, W) uint8 [0, 255] - already resized by decord
        video_tensor = torch.from_numpy(seq.egoview_RGB).float() / 255.0  # (T, 3, H, W)

        # Apply augmentation if enabled
        if self.data_aug:
            video_tensor = self._apply_video_augmentation(video_tensor)

        # Calculate the change in central pupil frame from frame i to frame i+1 for i = 0, ... , T-1.
        # T_{world, cpf}[i] is the transform from world to CPF at timestep i
        # We want T_{cpf[i], cpf[i+1]} = T_{world, cpf[i]}^{-1} @ T_{world, cpf[i+1]}
        # This gives us the relative motion in the local CPF frame.

        # Get CPF poses: rotation (T, 3, 3) and translation (T, 3)
        cpf_rotation = torch.from_numpy(seq.cpf_orientation).float()  # (T, 3, 3)
        cpf_translation = torch.from_numpy(seq.cpf_translation).float()  # (T, 3)

        # Create SE3 transform objects for each timestep.
        cpf_so3 = SO3.from_matrix(cpf_rotation)  # SO3 with batch shape (T,)
        T_world_cpf = SE3.from_rotation_and_translation(cpf_so3, cpf_translation)  # SE3 with batch shape (T,)

        # Compute relative transforms: T_{cpf[i], cpf[i+1]} for i = 0, ..., T-2
        # T_prev = T_{world, cpf}[:-1], T_curr = T_{world, cpf}[1:]
        T_world_cpf_prev = SE3(wxyz_xyz=T_world_cpf.wxyz_xyz[:-1])  # (T-1,)
        T_world_cpf_curr = SE3(wxyz_xyz=T_world_cpf.wxyz_xyz[1:])   # (T-1,)

        # T_{cpf[i], cpf[i+1]} = T_{world, cpf[i]}^{-1} @ T_{world, cpf[i+1]}
        T_cpf_change = T_world_cpf_prev.inverse() @ T_world_cpf_curr  # SE3 with batch shape (T-1,)

        # Convert to 6D twist representation (tangent space): (vx, vy, vz, omega_x, omega_y, omega_z)
        cpf_change_twist = T_cpf_change.log()  # (T-1, 6)
        # Pad first timestep with zeros (no previous frame to compare to)
        cpf_change_twist = torch.cat([
            torch.zeros(1, 6, dtype=torch.float32),
            cpf_change_twist
        ], dim=0)  # (T, 6)

        # Calculate the transformation of each joint in frame i for i = 0, ..., T-1.
        # T_{cpf, joint}[i] = transformation from the central pupil frame to the joint frame at timestep i.
        # T_{cpf, joint}[i] = T_{world, cpf}^-1 @ T_{world, joint}

        # Get the T_{world, joint} poses for each timestep of the nymeria sequence.
        joint_rotation = torch.from_numpy(seq.joint_orientation).float()    # (T, 23, 3, 3)
        joint_translation = torch.from_numpy(seq.joint_translation).float() # (T, 23, 3)

        # Create SE3 transform objects for each joint at each timestep.
        joint_so3 = SO3.from_matrix(joint_rotation)                                     # SO3 with batch shape (T, 23)
        T_world_joint = SE3.from_rotation_and_translation(joint_so3, joint_translation) # SE3 with batch shape (T, 23)

        # Compute T_cpf_joint = T_world_cpf^-1 @ T_world_joint
        # Add dimension for broadcasting: (T, 7) -> (T, 1, 7) to broadcast with (T, 23, 7)
        T_cpf_world = SE3(wxyz_xyz=T_world_cpf.wxyz_xyz.unsqueeze(1)).inverse()  # SE3 with batch shape (T, 1)
        T_cpf_joint = T_cpf_world @ T_world_joint  # SE3 with batch shape (T, 23)

        # Convert to twist representation and flatten (6D tangent space): (vx, vy, vz, omega_x, omega_y, omega_z)
        joint_twists = T_cpf_joint.log()  # (T, 23, 6)
        joint_twists_flat = joint_twists.reshape(joint_twists.shape[0], -1)  # (T, 23*6=138)

        # Concatenate CPF change twist (T, 6) with joint twists (T, 138) -> (T, 144)
        actions = torch.cat([cpf_change_twist, joint_twists_flat], dim=-1)  # (T, 6 + 23*6 = 144)

        data = {
            "obs": {
                "image": video_tensor,  # (T, 3, H, W) in [0, 1]
            },
            "action": actions,  # (T, 144) - CPF change twist (6) + joint twists (23*6=138)
        }

        # Add precomputed text embedding if available
        if self.text_embeddings is not None:
            # Get the filename for this sequence # Handle both regular dataset and Subset wrapper
            if hasattr(self.nymeria_dataset, 'hdf5_paths'): # Regular NymeriaDataset from pip-installed nymeria package
                hdf5_path = Path(self.nymeria_dataset.hdf5_paths[dataset_idx])
            else:
                # Subset wrapper - access underlying dataset
                underlying_idx = self.nymeria_dataset.indices[dataset_idx]
                hdf5_path = Path(self.nymeria_dataset.dataset.hdf5_paths[underlying_idx])

            filename = hdf5_path.name
            if filename in self.text_embeddings:
                # Add as language_latents for direct use by the model
                data["language_latents"] = self.text_embeddings[filename]  # (512,)
            else:
                # Fallback: zero embedding if text not found
                print(f"WARNING: No text embedding found for {filename}")
                data["language_latents"] = torch.zeros(512)

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
