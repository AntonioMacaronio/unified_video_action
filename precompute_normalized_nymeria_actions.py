"""
Script to precompute the normalizer for the Nymeria dataset.
This avoids the slow normalizer computation during distributed training.

Usage:
    python precompute_normalized_nymeria_actions.py --data_dir /data/nymeria/ --output normalizer.pkl
"""

import argparse
import pickle
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_video_action.dataset.nymeria_dataset import NymeriaUVADataset


def main():
    parser = argparse.ArgumentParser(description="Precompute normalizer for Nymeria dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to nymeria HDF5 data directory")
    parser.add_argument("--output", type=str, default="normalizer.pkl", help="Output path for normalizer pickle file")
    parser.add_argument("--text_embeddings_path", type=str, default=None, help="Path to precomputed text embeddings (optional)")
    parser.add_argument("--sequence_length", type=int, default=144, help="Sequence length")
    parser.add_argument("--image_resolution", type=int, default=256, help="Image resolution")
    parser.add_argument("--val_ratio", type=float, default=0.02, help="Validation ratio")
    parser.add_argument("--normalizer_type", type=str, default="limits", help="Normalizer type (limits or gaussian)")
    args = parser.parse_args()

    print(f"Initializing NymeriaUVADataset from {args.data_dir}...")

    dataset = NymeriaUVADataset(
        data_dir=args.data_dir,
        image_resolution=args.image_resolution,
        sequence_length=args.sequence_length,
        val_ratio=args.val_ratio,
        seed=42,
        data_aug=False,  # No augmentation needed for normalizer computation
        file_pattern="*.h5",
        language_emb_model="clip",
        normalizer_type=args.normalizer_type,
        num_datapoints=-1,
        text_embeddings_path=args.text_embeddings_path,
    )

    print(f"Dataset initialized with {len(dataset)} training samples")
    print(f"Computing normalizer (mode={args.normalizer_type})...")

    normalizer = dataset.get_normalizer(mode=args.normalizer_type)

    print(f"Saving normalizer to {args.output}...")
    with open(args.output, "wb") as f:
        pickle.dump(normalizer, f)

    print(f"Done! Normalizer saved to {args.output}")
    print(f"\nTo use this normalizer, add to your training command:")
    print(f"  task.precomputed_normalizer_path={os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
