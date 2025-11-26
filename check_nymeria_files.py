#!/usr/bin/env python3
"""
Diagnostic script to check all Nymeria HDF5 files for invalid num_frames.

NOTE: Please run this script on the SSH'ed GPU cluster provisioned by SkyPilot.
This script won't work on your local metastream.
"""

import h5py
import hdf5plugin  # Required for reading LZ4-compressed HDF5 files
from pathlib import Path
import sys

def check_hdf5_file(filepath):
    """Check a single HDF5 file for issues."""
    try:
        with h5py.File(filepath, 'r') as f:
            num_frames = f.attrs.get('num_frames', None)
            sequence_name = f.attrs.get('sequence_name', 'UNKNOWN')

            # Get actual array length
            actual_length = len(f['timestamp_ns'][:]) if 'timestamp_ns' in f else 0

            if num_frames is None:
                print(f"WARNING: {filepath.name}")
                print(f"  num_frames attribute is MISSING")
                print(f"  sequence_name: {sequence_name}")
                print(f"  actual array length: {actual_length}")
                print()
                return False

            if num_frames < 0:
                print(f"ERROR: {filepath.name}")
                print(f"  num_frames: {num_frames} (NEGATIVE!)")
                print(f"  sequence_name: {sequence_name}")
                print(f"  actual array length: {actual_length}")
                print()
                return False

            if num_frames != actual_length:
                print(f"WARNING: {filepath.name}")
                print(f"  num_frames: {num_frames}")
                print(f"  actual array length: {actual_length} (MISMATCH!)")
                print(f"  sequence_name: {sequence_name}")
                print()
                return False

            return True
    except Exception as e:
        print(f"ERROR reading {filepath.name}: {e}")
        return False

def main():
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        data_dir = Path("/nfs/antzhan/nymeria/hdf5")

    if not data_dir.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        sys.exit(1)

    print(f"Checking HDF5 files in: {data_dir}")
    print("=" * 80)
    print()

    hdf5_files = sorted(list(data_dir.glob("*.h5")))

    if not hdf5_files:
        print(f"No .h5 files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(hdf5_files)} HDF5 files")
    print()

    good_files = 0
    bad_files = 0

    for filepath in hdf5_files:
        if check_hdf5_file(filepath):
            good_files += 1
        else:
            bad_files += 1

    print("=" * 80)
    print(f"Summary:")
    print(f"  Total files: {len(hdf5_files)}")
    print(f"  Good files: {good_files}")
    print(f"  Bad files: {bad_files}")

    if bad_files > 0:
        print()
        print(f"Found {bad_files} problematic files! See details above.")
        sys.exit(1)
    else:
        print()
        print("All files look good!")

if __name__ == "__main__":
    main()
