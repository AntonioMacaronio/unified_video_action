#!/usr/bin/env python3
"""Find empty parameters in the model that break DeepSpeed ZeRO-3."""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import torch
import hydra
from omegaconf import OmegaConf
import pathlib
import sys

# Set up Hydra config path
sys.argv = [
    'train.py',
    '--config-dir=.',
    '--config-name=uva_nymeria.yaml',
    'model.policy.action_model_params.predict_action=False',
    'model.policy.selected_training_mode=video_model',
    'task.dataset.data_dir=/nfs/antzhan/nymeria/hdf5',
]

# Initialize Hydra
hydra.initialize_config_dir(
    config_dir=str(pathlib.Path('.').absolute() / "unified_video_action" / "config"),
    version_base=None,
)
cfg = hydra.compose(
    config_name="uva_nymeria.yaml",
    overrides=[
        'model.policy.action_model_params.predict_action=False',
        'model.policy.selected_training_mode=video_model',
    ],
)

print("Creating model...")
from unified_video_action.workspace.train_unified_video_action_workspace import TrainUnifiedVideoActionWorkspace

workspace = TrainUnifiedVideoActionWorkspace(cfg)

print("\n" + "=" * 80)
print("Checking all model parameters...")
print("=" * 80)

empty_params = []
for name, param in workspace.model.named_parameters():
    if param.numel() == 0:
        empty_params.append((name, param.shape, param.requires_grad))
        print(f"❌ EMPTY PARAMETER: {name}")
        print(f"   Shape: {param.shape}, numel: {param.numel()}, requires_grad: {param.requires_grad}")

for name, buffer in workspace.model.named_buffers():
    if buffer.numel() == 0:
        empty_params.append((name, buffer.shape, "buffer"))
        print(f"❌ EMPTY BUFFER: {name}")
        print(f"   Shape: {buffer.shape}, numel: {buffer.numel()}")

if empty_params:
    print("\n" + "=" * 80)
    print(f"FOUND {len(empty_params)} EMPTY PARAMETERS/BUFFERS!")
    print("These will break DeepSpeed ZeRO-3")
    print("=" * 80)
else:
    print("\n✅ No empty parameters found")