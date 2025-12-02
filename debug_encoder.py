"""Debug script to check model configuration values."""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import torch
import hydra
from omegaconf import OmegaConf
import pathlib

hydra.initialize_config_dir(
    config_dir=str(pathlib.Path('.').absolute() / "unified_video_action" / "config"),
    version_base=None,
)

cfg = hydra.compose(
    config_name="uva_nymeria.yaml",
    overrides=[
        'model.policy.action_model_params.predict_action=False',
        'model.policy.selected_training_mode=video_model',
        'task.dataset.data_dir=/nfs/antzhan/nymeria/hdf5',
    ],
)

print("=" * 80)
print("Configuration Values:")
print("=" * 80)

policy_cfg = cfg.model.policy

flags = {
    'use_history_action': policy_cfg.get('use_history_action'),
    'use_proprioception': policy_cfg.get('use_proprioception'),
    'predict_wrist_img': policy_cfg.get('predict_wrist_img'),
    'predict_proprioception': policy_cfg.get('predict_proprioception'),
    'task_name': cfg.task.name,
    'action_dim': cfg.task.shape_meta.action.shape[0],
}

for key, value in flags.items():
    print(f"{key:25} = {value}")

# Calculate expected proj_cond_x_dim_num
predict_wrist_img = flags['predict_wrist_img']
use_proprioception = flags['use_proprioception']
use_history_action = flags['use_history_action']
task_name = flags['task_name']

if predict_wrist_img:
    proj_cond_x_dim_num = 4
    if use_proprioception:
        proj_cond_x_dim_num += 2
    if use_history_action:
        proj_cond_x_dim_num += 1
else:
    proj_cond_x_dim_num = 3
    if use_proprioception:
        if task_name == "umi" or "block_push" in task_name or "pusht" in task_name:
            proj_cond_x_dim_num += 1
        else:
            proj_cond_x_dim_num += 2
    if use_history_action:
        proj_cond_x_dim_num += 1

print("\n" + "=" * 80)
print(f"Expected proj_cond_x_dim_num: {proj_cond_x_dim_num}")
print(f"Expected input dim to proj_cond_x_layer: {proj_cond_x_dim_num} * encoder_embed_dim")
print("=" * 80)

# Now check what the pretrained model has
print("\nLoading pretrained model to check its config...")
pretrained_path = 'pretrained_models/mar/mar_base/checkpoint-last.pth'
if os.path.exists(pretrained_path):
    ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)

    # Check if model state has the proj_cond_x_layer
    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    proj_layer_key = 'proj_cond_x_layer.weight'
    if proj_layer_key in state_dict:
        weight_shape = state_dict[proj_layer_key].shape
        print(f"Pretrained {proj_layer_key} shape: {weight_shape}")
        print(f"  Input dim: {weight_shape[1]}")
        print(f"  Output dim: {weight_shape[0]}")
        encoder_embed_dim = weight_shape[0]
        pretrained_proj_dim = weight_shape[1] // encoder_embed_dim
        print(f"  Pretrained proj_cond_x_dim_num: {pretrained_proj_dim}")
        print(f"  encoder_embed_dim: {encoder_embed_dim}")

        if pretrained_proj_dim != proj_cond_x_dim_num:
            print("\n" + "!" * 80)
            print(f"MISMATCH FOUND!")
            print(f"  Pretrained model expects: {pretrained_proj_dim} parts")
            print(f"  Your config will create: {proj_cond_x_dim_num} parts")
            print("!" * 80)
    else:
        print(f"Warning: {proj_layer_key} not found in checkpoint")
else:
    print(f"Pretrained model not found at: {pretrained_path}")