"""Profiling script to identify dataloading bottlenecks.

Run with:
python dataloading_profile.py
"""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import h5py
import hdf5plugin
from decord import VideoReader, cpu
import torchvision.transforms as transforms

# Adjust this path to your data directory
DATA_DIR = "/nfs/antzhan/nymeria/mp4"


def profile_single_item_detailed():
    """Profile a single item load with detailed breakdown."""
    print("\n" + "=" * 60)
    print("DETAILED PROFILING: Single Item Breakdown")
    print("=" * 60)

    from nymeria import NymeriaDataset, NymeriaTrainingSeq

    # Get first file
    data_dir = Path(DATA_DIR)
    hdf5_files = sorted(list(data_dir.glob("*.h5")))
    hdf5_path = hdf5_files[10]
    mp4_path = hdf5_path.with_suffix('.mp4')

    print(f"\nProfiling file: {hdf5_path.name}")
    print(f"MP4 file size: {mp4_path.stat().st_size / 1024 / 1024:.2f} MB")

    # 1. HDF5 loading only
    t0 = time.time()
    with h5py.File(hdf5_path, 'r') as f:
        timestamp_ns = f['timestamp_ns'][:]
        root_translation = f['root_translation'][:]
        root_orientation = f['root_orientation'][:]
        cpf_translation = f['cpf_translation'][:]
        cpf_orientation = f['cpf_orientation'][:]
        joint_translation = f['joint_translation'][:]
        joint_orientation = f['joint_orientation'][:]
        contact_information = f['contact_information'][:]
    hdf5_time = time.time() - t0
    print(f"\n[1] HDF5 full loading:         {hdf5_time:.3f}s")

    # 2. MP4 decoding with decord (full resolution)
    t0 = time.time()
    vr = VideoReader(str(mp4_path), ctx=cpu(0))
    videoreader_init_time = time.time() - t0
    print(f"[2] VideoReader init:          {videoreader_init_time:.3f}s")

    t0 = time.time()
    video_frames = vr[:].asnumpy()  # (N, H, W, C)
    decode_time = time.time() - t0
    print(f"[3] MP4 decode (full res):     {decode_time:.3f}s")
    print(f"    Decoded shape: {video_frames.shape}, dtype: {video_frames.dtype}")

    # 4. MP4 decoding with resize DURING decode (decord handles resize)
    image_resolution = 256
    t0 = time.time()
    vr_resized = VideoReader(str(mp4_path), ctx=cpu(0), width=image_resolution, height=image_resolution)
    frames_decoded_resized = vr_resized[:].asnumpy()  # (N, H, W, C) already at 256x256
    decode_resize_time = time.time() - t0
    print(f"[4] MP4 decode+resize (256):   {decode_resize_time:.3f}s")
    print(f"    Decoded shape: {frames_decoded_resized.shape}")

    # 5. Transpose to (N, C, H, W)
    t0 = time.time()
    egoview_RGB = np.transpose(video_frames, (0, 3, 1, 2))
    transpose_time = time.time() - t0
    print(f"[5] Numpy transpose:           {transpose_time:.3f}s")

    # 6. Test NymeriaTrainingSeq full load (what the dataset actually does)
    t0 = time.time()
    seq = NymeriaTrainingSeq(hdf5_path, mp4_path)
    full_seq_load_time = time.time() - t0
    print(f"[6] NymeriaTrainingSeq load:   {full_seq_load_time:.3f}s (HDF5 + MP4 combined)")

    # 7. pad_or_trim_sequence
    sequence_length = 150
    t0 = time.time()
    seq_trimmed, padding_mask = seq.pad_or_trim_sequence(sequence_length)
    pad_trim_time = time.time() - t0
    print(f"[7] pad_or_trim_sequence:      {pad_trim_time:.3f}s")
    print(f"    Original len: {len(seq)}, Trimmed len: {len(seq_trimmed)}")

    # 8. Frame-by-frame processing (current implementation)
    image_resolution = 256
    resize_transform = transforms.Resize((image_resolution, image_resolution), antialias=True)

    t0 = time.time()
    processed_frames = []
    for t in range(sequence_length):
        frame = torch.from_numpy(seq_trimmed.egoview_RGB[t]) / np.float32(255)
        frame = resize_transform(frame)
        processed_frames.append(frame)
    video_tensor = torch.stack(processed_frames)
    frame_loop_time = time.time() - t0
    print(f"[8] Frame-by-frame loop (150): {frame_loop_time:.3f}s")
    print(f"    Output shape: {video_tensor.shape}")

    # 9. Alternative: Batched processing with F.interpolate
    t0 = time.time()
    frames_np = seq_trimmed.egoview_RGB[:sequence_length]  # (T, C, H, W)
    video_tensor_v2 = torch.from_numpy(frames_np).float() / 255.0
    video_tensor_v2 = torch.nn.functional.interpolate(
        video_tensor_v2,
        size=(image_resolution, image_resolution),
        mode='bilinear',
        align_corners=False
    )
    batched_interp_time = time.time() - t0
    print(f"[9] Batched F.interpolate:     {batched_interp_time:.3f}s")

    # 10. Alternative: Batched torchvision resize
    t0 = time.time()
    frames_np = seq_trimmed.egoview_RGB[:sequence_length]  # (T, C, H, W)
    video_tensor_v3 = torch.from_numpy(frames_np).float() / 255.0
    video_tensor_v3 = resize_transform(video_tensor_v3)  # torchvision handles batch
    batched_tv_time = time.time() - t0
    print(f"[10] Batched torchvision:      {batched_tv_time:.3f}s")
    print(f"     Output shape: {video_tensor_v3.shape}")

    # Summary
    total_current = full_seq_load_time + pad_trim_time + frame_loop_time
    # Optimized path: decode+resize in one step, minimal post-processing
    total_optimized = decode_resize_time + 0.01  # Just transpose + normalize

    print(f"\n{'='*40}")
    print(f"SUMMARY (single item):")
    print(f"  NymeriaTrainingSeq load: {full_seq_load_time:.3f}s ({full_seq_load_time/total_current*100:.0f}%)")
    print(f"  pad_or_trim_sequence:    {pad_trim_time:.3f}s ({pad_trim_time/total_current*100:.0f}%)")
    print(f"  Frame processing:        {frame_loop_time:.3f}s ({frame_loop_time/total_current*100:.0f}%)")
    print(f"  ----------------------------------------")
    print(f"  Current total:           {total_current:.3f}s")
    print(f"{'='*40}")
    print(f"\nPOTENTIAL OPTIMIZATION:")
    print(f"  Current (decode full + resize after): {decode_time + frame_loop_time:.3f}s")
    print(f"  Optimized (decode+resize together):   {decode_resize_time:.3f}s")
    print(f"  Potential speedup:                    {(decode_time + frame_loop_time)/decode_resize_time:.1f}x")
    print(f"{'='*40}")

    return {
        'hdf5': hdf5_time,
        'vr_init': videoreader_init_time,
        'decode': decode_time,
        'decode_resize': decode_resize_time,
        'transpose': transpose_time,
        'full_seq_load': full_seq_load_time,
        'pad_trim': pad_trim_time,
        'frame_loop': frame_loop_time,
        'batched_interp': batched_interp_time,
        'batched_tv': batched_tv_time,
    }


def profile_batch(batch_size=16):
    """Profile loading a full batch."""
    print("\n" + "=" * 60)
    print(f"PROFILING: Full Batch (batch_size={batch_size})")
    print("=" * 60)

    from unified_video_action.dataset.nymeria_dataset import NymeriaUVADataset

    dataset = NymeriaUVADataset(
        data_dir=DATA_DIR,
        image_resolution=256,
        sequence_length=150,
        val_ratio=0.02,
        seed=42,
    )

    # Profile loading batch_size items sequentially (simulates num_workers=0)
    print(f"\nLoading {batch_size} items sequentially...")

    item_times = []
    for i in range(batch_size):
        t0 = time.time()
        item = dataset[i]
        elapsed = time.time() - t0
        item_times.append(elapsed)
        print(f"  Item {i:2d}: {elapsed:.2f}s")

    total_time = sum(item_times)
    avg_time = np.mean(item_times)

    print(f"\n{'='*40}")
    print(f"BATCH SUMMARY:")
    print(f"  Total time:      {total_time:.2f}s")
    print(f"  Avg per item:    {avg_time:.2f}s")
    print(f"  Min/Max:         {min(item_times):.2f}s / {max(item_times):.2f}s")
    print(f"{'='*40}")


def main():
    print("=" * 60)
    print("Nymeria DataLoader Profiling Script")
    print("=" * 60)

    # Detailed single-item profiling
    profile_single_item_detailed()

    # Full batch profiling
    profile_batch(batch_size=16)

    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
