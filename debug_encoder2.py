"""Inspect pretrained checkpoint structure."""

import torch

pretrained_path = 'pretrained_models/mar/mar_base/checkpoint-last.pth'

print("=" * 80)
print(f"Loading: {pretrained_path}")
print("=" * 80)

ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)

print("\nTop-level keys:")
for key in ckpt.keys():
    print(f"  - {key}")

# Find the model state dict
if 'model' in ckpt:
    state_dict = ckpt['model']
    dict_name = 'model'
elif 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
    dict_name = 'state_dict'
elif 'ema_model' in ckpt:
    state_dict = ckpt['ema_model']
    dict_name = 'ema_model'
else:
    state_dict = ckpt
    dict_name = '(root)'

print(f"\nUsing state dict from: {dict_name}")
print(f"Number of parameters: {len(state_dict)}")

# Look for relevant keys
print("\nSearching for proj/embedding layers...")
relevant_keys = []
for key in state_dict.keys():
    if any(x in key for x in ['proj', 'embed', 'cond', 'history', 'action', 'proprioception']):
        relevant_keys.append(key)

relevant_keys.sort()
for key in relevant_keys[:50]:  # First 50 matches
    shape = state_dict[key].shape
    print(f"  {key:60} {str(shape):30}")

# Check if there's config info
if 'config' in ckpt:
    print("\n" + "=" * 80)
    print("Config found in checkpoint:")
    print(ckpt['config'])

#!/usr/bin/env python3
"""Compare model architecture with different flag settings."""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import torch
import hydra
from omegaconf import OmegaConf
import pathlib

# Test different combinations
configs_to_test = [
    {
        'name': 'Current (Nymeria)',
        'config': 'uva_nymeria.yaml',
        'overrides': [],
    },
    {
        'name': 'Reference (PushT)',
        'config': 'uva_pusht.yaml',
        'overrides': [],
    },
    {
        'name': 'Nymeria with explicit False flags',
        'config': 'uva_nymeria.yaml',
        'overrides': [
            'model.policy.use_history_action=false',
            'model.policy.use_proprioception=false',
            'model.policy.predict_wrist_img=false',
            'model.policy.predict_proprioception=false',
        ],
    },
]

for test in configs_to_test:
    print("\n" + "=" * 80)
    print(f"Testing: {test['name']}")
    print("=" * 80)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(
        config_dir=str(pathlib.Path('.').absolute() / "unified_video_action" / "config"),
        version_base=None,
    )

    base_overrides = [
        'model.policy.action_model_params.predict_action=False',
        'model.policy.selected_training_mode=video_model',
        'task.dataset.data_dir=/nfs/antzhan/nymeria/hdf5',  # dummy, won't be used
    ]

    try:
        cfg = hydra.compose(
            config_name=test['config'],
            overrides=base_overrides + test['overrides'],
        )

        # Extract flags
        policy = cfg.model.policy
        flags = {
            'use_history_action': policy.get('use_history_action'),
            'use_proprioception': policy.get('use_proprioception'),
            'predict_wrist_img': policy.get('predict_wrist_img'),
            'predict_proprioception': policy.get('predict_proprioception'),
            'task_name': cfg.task.name,
        }

        print("\nFlags:")
        for k, v in flags.items():
            print(f"  {k:25} = {v}")

        # Calculate proj_cond_x_dim_num
        use_history_action = flags['use_history_action']
        use_proprioception = flags['use_proprioception']
        predict_wrist_img = flags['predict_wrist_img']
        task_name = flags['task_name']

        if predict_wrist_img:
            proj_cond_x_dim_num = 4
            if use_proprioception:
                proj_cond_x_dim_num += 2
            if use_history_action:
                proj_cond_x_dim_num += 1
        else:
            proj_cond_x_dim_num = 3  # [x, cond, action_latents]
            if use_proprioception:
                if task_name in ["umi", "block_push", "pusht"]:
                    proj_cond_x_dim_num += 1
                else:
                    proj_cond_x_dim_num += 2
            if use_history_action:
                proj_cond_x_dim_num += 1

        print(f"\nCalculated proj_cond_x_dim_num: {proj_cond_x_dim_num}")
        print(f"Parts being concatenated: {proj_cond_x_dim_num}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()