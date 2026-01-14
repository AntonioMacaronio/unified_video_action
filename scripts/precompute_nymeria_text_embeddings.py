"""
Precompute CLIP text embeddings for Nymeria dataset atomic_action descriptions.

This script iterates through all HDF5 files in the Nymeria dataset, extracts the
atomic_action text from each file, computes CLIP text embeddings, and saves them
to a pickle file for fast loading during training.

Usage:
    python scripts/precompute_nymeria_text_embeddings.py \
        --data-dir /nfs/antzhan/nymeria/hdf5 \
        --output-path /nfs/antzhan/nymeria/hdf5/text_embeddings.pkl

The output pickle file contains a dictionary:
    {
        "embeddings": {filename: embedding_tensor},  # filename -> (512,) tensor
        "texts": {filename: atomic_action_text},      # filename -> str
        "model": "openai/clip-vit-base-patch32"
    }
"""

from dataclasses import dataclass
from pathlib import Path
import pickle
import h5py
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, CLIPModel
import tyro


@dataclass
class Args:
    """Precompute CLIP text embeddings for Nymeria dataset."""

    data_dir: str = "/data/nymeria/"
    """Path to Nymeria HDF5 directory"""

    output_path: str | None = None
    """Output pickle file path (default: data_dir/text_embeddings.pkl)"""

    batch_size: int = 64
    """Batch size for CLIP encoding"""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Device to run CLIP model on"""


def extract_atomic_action_from_hdf5(hdf5_path: Path) -> str:
    """Extract the atomic_action string from an HDF5 file."""
    with h5py.File(hdf5_path, 'r') as f:
        atomic_action = f.attrs.get('atomic_action', '')
        if isinstance(atomic_action, bytes):
            atomic_action = atomic_action.decode('utf-8')
    return atomic_action


def main(args: Args):
    data_dir = Path(args.data_dir)
    output_path = Path(args.output_path) if args.output_path else data_dir / "text_embeddings.pkl"

    # Find all HDF5 files
    hdf5_files = sorted(list(data_dir.glob("*.h5")))
    print(f"Found {len(hdf5_files)} HDF5 files in {data_dir}")

    if len(hdf5_files) == 0:
        print("No HDF5 files found. Exiting.")
        return

    # Load CLIP model
    print("Loading CLIP model...")
    model_name = "openai/clip-vit-base-patch32"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(args.device)
    model.eval()

    # Extract all atomic_action texts
    print("Extracting atomic_action texts from HDF5 files...")
    texts = {}
    for hdf5_path in tqdm(hdf5_files, desc="Reading HDF5 files"):
        filename = hdf5_path.name
        atomic_action = extract_atomic_action_from_hdf5(hdf5_path)
        texts[filename] = atomic_action

    # Get unique texts to avoid redundant computation
    unique_texts = list(set(texts.values()))
    print(f"Found {len(unique_texts)} unique atomic_action descriptions")

    # Compute embeddings for unique texts in batches
    print("Computing CLIP embeddings...")
    unique_embeddings = {}

    with torch.no_grad():
        for i in tqdm(range(0, len(unique_texts), args.batch_size), desc="Encoding texts"):
            batch_texts = unique_texts[i:i + args.batch_size] # list of strings
            # ex: ['While standing and leaning to the left in the living area, C measures the height of the table with a roll meter in her right hand.', 'C is sitting on the sofa in the living area and holding a remote control in both of her hands.']

            # Tokenize
            tokens = tokenizer(
                batch_texts,
                padding="max_length",
                max_length=77,  # CLIP's max context length
                truncation=True,
                return_tensors="pt"
            ).to(args.device) # <transformers.tokenization_utils_base.BatchEncoding> object
            # tokens is a dictionary with keys: 'input_ids', 'attention_mask'
            # tokens['input_ids'] is a tensor of shape (batch_size, max_length)
            #   - each element is an integer that enumerates a token in the vocabulary
            # tokens['attention_mask'] is a tensor of shape (batch_size, max_length)
            #   - tokenizer is forcing every sentence to be the same length (77 tokens)
            #   - this means some tokens are just padding tokens, which is the purpose of the attention mask.

            # For each sentence, convert the 77 tokens into a 512-dimensional vector using the CLIP text encoder.
            text_features = model.get_text_features(**tokens)  # (batch_size, 512)

            # Add embeddings to a dictionary of {text: (512,) tensor}
            for j, text in enumerate(batch_texts):
                unique_embeddings[text] = text_features[j].cpu()

    # Map filenames to embeddings (this is a dictionary of {filename: (512,) tensor})
    print("Creating filename -> embedding mapping...")
    embeddings = {} 
    for filename, text in texts.items():
        embeddings[filename] = unique_embeddings[text]

    # Save to pickle
    output_data = {
        "embeddings": embeddings,
        "texts": texts,
        "model": model_name,
        "embedding_dim": 512,
    }

    print(f"Saving embeddings to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)

    print(f"Done! Saved {len(embeddings)} embeddings to {output_path}")
    print(f"  - Embedding dimension: 512")
    print(f"  - Unique texts: {len(unique_texts)}")
    print(f"  - Total files: {len(embeddings)}")

    # Print some example texts
    print("\nExample atomic_action texts:")
    for i, (filename, text) in enumerate(list(texts.items())[:5]):
        print(f"  {filename}: {text[:80]}...")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
