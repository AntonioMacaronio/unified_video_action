"""Debug script for multi-GPU training hang - run with accelerate launch."""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import time
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from accelerate import Accelerator

DATA_DIR = "/nfs/antzhan/nymeria/hdf5"

def log(accelerator, msg):
    """Print with rank prefix and flush immediately."""
    print(f"[Rank {accelerator.process_index}/{accelerator.num_processes}] {msg}", flush=True)

def main():
    # Step 1: Initialize accelerator
    print("Initializing accelerator...", flush=True)
    accelerator = Accelerator()
    log(accelerator, "Accelerator initialized")

    # Step 2: Test barrier
    log(accelerator, "Testing barrier...")
    accelerator.wait_for_everyone()
    log(accelerator, "Barrier passed")

    # Step 3: Create dataset
    log(accelerator, "Creating dataset...")
    t0 = time.time()
    from unified_video_action.dataset.nymeria_dataset import NymeriaUVADataset
    dataset = NymeriaUVADataset(
        data_dir=DATA_DIR,
        image_resolution=224,
        sequence_length=150,
        val_ratio=0.02,
        seed=42,
    )
    log(accelerator, f"Dataset created in {time.time()-t0:.2f}s, len={len(dataset)}")

    # Step 4: Create dataloader
    log(accelerator, "Creating dataloader...")
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0, shuffle=True)
    log(accelerator, f"DataLoader created, len={len(dataloader)}")

    # Step 5: Prepare with accelerator (adds DistributedSampler)
    log(accelerator, "Calling accelerator.prepare()...")
    t0 = time.time()
    dataloader = accelerator.prepare(dataloader)
    log(accelerator, f"accelerator.prepare() done in {time.time()-t0:.2f}s")

    # Step 6: Test barrier after prepare
    log(accelerator, "Testing barrier after prepare...")
    accelerator.wait_for_everyone()
    log(accelerator, "Barrier passed")

    # Step 7: Load first batch
    log(accelerator, "Loading first batch...")
    t0 = time.time()
    batch = next(iter(dataloader))
    log(accelerator, f"First batch loaded in {time.time()-t0:.2f}s, shape={batch['obs']['image'].shape}")

    # Step 8: Test barrier after data load
    log(accelerator, "Testing barrier after data load...")
    accelerator.wait_for_everyone()
    log(accelerator, "Barrier passed")

    # Step 9: Test a simple all-reduce (simulates gradient sync)
    log(accelerator, "Testing NCCL all-reduce...")
    t0 = time.time()
    tensor = torch.ones(1000, 1000, device=accelerator.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    log(accelerator, f"All-reduce done in {time.time()-t0:.4f}s, sum={tensor[0,0].item()}")

    # Step 10: Final barrier
    accelerator.wait_for_everyone()
    log(accelerator, "SUCCESS - All tests passed!")

if __name__ == "__main__":
    main()