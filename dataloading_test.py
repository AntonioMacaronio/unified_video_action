"""Debug script to test Nymeria dataloader with single process.

Run with
python dataloading_test.py
"""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import time
import torch
from torch.utils.data import DataLoader

# Adjust this path to your data directory
DATA_DIR = "/data/nymeria"

def main():
    print("=" * 60)
    print("Nymeria DataLoader Debug Script")
    print("=" * 60)

    # Step 1: Test raw NymeriaDataset from pip package
    print("\n[1] Testing raw NymeriaDataset (pip package)...")
    t0 = time.time()
    from nymeria import NymeriaDataset as RawNymeriaDataset
    raw_dataset = RawNymeriaDataset(DATA_DIR, file_pattern="*.h5")
    print(f"    Dataset created in {time.time() - t0:.2f}s, length: {len(raw_dataset)}")

    print("\n[2] Testing single item access (raw)...")
    t0 = time.time()
    item = raw_dataset[0]
    print(f"    Item 0 loaded in {time.time() - t0:.2f}s")
    print(f"    Type: {type(item)}")
    if hasattr(item, 'egoview_RGB'):
        print(f"    egoview_RGB shape: {item.egoview_RGB.shape}")

    # Step 2: Test NymeriaUVADataset wrapper
    print("\n[3] Testing NymeriaUVADataset wrapper...")
    t0 = time.time()
    from unified_video_action.dataset.nymeria_dataset import NymeriaUVADataset
    dataset = NymeriaUVADataset(
        data_dir=DATA_DIR,
        image_resolution=224,
        sequence_length=150,
        val_ratio=0.02,
        seed=42,
    )
    print(f"    UVA Dataset created in {time.time() - t0:.2f}s, length: {len(dataset)}")

    print("\n[4] Testing single item access (UVA wrapper)...")
    t0 = time.time()
    item = dataset[0]
    print(f"    Item 0 loaded in {time.time() - t0:.2f}s")
    print(f"    Keys: {item.keys()}")
    if 'obs' in item and 'image' in item['obs']:
        print(f"    obs/image shape: {item['obs']['image'].shape}")

    # Step 3: Test DataLoader
    print("\n[5] Testing DataLoader (num_workers=0, batch_size=16)...")
    dataloader = DataLoader(dataset, batch_size=16, num_workers=0, shuffle=True)

    print("\n[6] Iterating first 3 batches...")
    dataloader_iter = iter(dataloader)
    for i in range(3):
        t0 = time.time()
        batch = next(dataloader_iter)
        elapsed = time.time() - t0
        print(f"    Batch {i}: image shape = {batch['obs']['image'].shape}, loaded in {elapsed:.2f}s")

    print("\n" + "=" * 60)
    print("SUCCESS - DataLoader works with single process!")
    print("=" * 60)

if __name__ == "__main__":
    main()