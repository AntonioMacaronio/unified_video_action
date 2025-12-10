import sys

sys.path.extend([".", "src"])
import torch
import os
from einops import rearrange
import torch.nn.functional as F
import wandb

from unified_video_action.fvd.fvd import get_fvd_logits, frechet_distance
from unified_video_action.fvd.download import load_i3d_pretrained
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.utils.utils import AverageMeter
from unified_video_action.utils.data_utils import resize_image
from unified_video_action.utils.data_utils import (
    normalize_action,
    normalize_obs,
    unnormalize_future_action,
)
from unified_video_action.utils.data_utils import (
    process_data,
    save_image_grid,
    get_vae_latent,
    get_trajectory,
    decode_from_sample_autoregressive,
)
from unified_video_action.utils.language_model import extract_text_features
from unified_video_action.utils.data_utils import extract_latent_autoregressive



def prepare_data_predict_action(
    cfg, x, actions, model, T, device, language_goal=None, eval=False
):
    ## normalize actions and observations
    nactions = normalize_action(
        normalizer=model.normalizer,
        normalizer_type=model.normalizer_type,
        actions=actions,
    )
    x = normalize_obs(
        normalizer=model.normalizer, normalizer_type=model.normalizer_type, batch=x
    )

    ## process data
    x, proprioception_input, _ = process_data(
        x,
        task_name=cfg.task.name,
        eval=eval,
        use_proprioception=cfg.model.policy.use_proprioception,
        different_history_freq=cfg.model.policy.different_history_freq,
    )

    real, _, c, latent_size, proprioception_input = get_vae_latent(
        x, model.vae_model, eval=True, proprioception_input=proprioception_input
    )
    history_trajectory, trajectory = get_trajectory(
        nactions,
        T,
        cfg.model.policy.shift_action,  # true for uva_nymeria.yaml (it's inside uva.yaml)
        use_history_action=cfg.model.policy.use_history_action, # true for uva_nymeria.yaml
    )

    text_latents = None
    if cfg.task.dataset.language_emb_model is not None:
        if "umi" in cfg.task.name:
            text_latents = language_goal
        elif "libero" in cfg.task.name:
            if cfg.task.dataset.language_emb_model == "clip":
                text_tokens = {
                    "input_ids": language_goal[:, 0].long()[:, 0],
                    "attention_mask": language_goal[:, 0].long()[:, 1],
                }
                text_latents = extract_text_features(
                    model.text_model,
                    text_tokens,
                    language_emb_model=cfg.task.dataset.language_emb_model,
                )
            elif cfg.task.dataset.language_emb_model == "flant5":
                text_tokens = language_goal[:, 0].long()
                text_latents = extract_text_features(
                    model.text_model,
                    text_tokens,
                    language_emb_model=cfg.task.dataset.language_emb_model,
                ).float()
            else:
                raise NotImplementedError
    return (
        x,
        real,
        latent_size,
        c,
        text_latents,
        history_trajectory,
        trajectory,
        proprioception_input,
    )


def test_video_fvd(
    cfg, model, loader, it, output_dir, device, name_label="", plot_actions=False
):
    """
    # these are example values passed when this function is called from train_unified_video_action_workspace.py
    cfg = uva_nymeria.yaml 
    model = UnifiedVideoActionPolicy
    loader = Dataloader(val_dataset)
    it = local_epoch_idx
    output_dir = self.output_dir
    device = cuda
    """
    losses = dict()
    losses["fvd"] = AverageMeter()

    i3d = load_i3d_pretrained(device)
    real_embeddings = []
    pred_embeddings = []

    reals = []
    predictions = []

    n_examples = 4

    with torch.no_grad():
        for n, batch in enumerate(loader):
            if n % 10 == 0:
                print("test_video_fvd", n, len(loader))

            x = batch
            # `x` or `batch` is a dictionary with keys: 'obs', 'action', 'language', it has the following structure:
            # {
            #     'obs': {
            #         'image': (B, T, C, H, W)
            #     },
            #     'action': (B, T, 1) # dummy actions (all zeros)
            #     'language': (B, T, L) # language goal (will add this later)
            # }
            if n >= n_examples:
                break

            x = dict_apply(x, lambda x: x.to(device, non_blocking=True))
            actions = x["action"]

            if cfg.model.policy.use_history_action:
                x = dict_apply(x, lambda x: x[:, 1:])

            x = resize_image(cfg, x)

            B, T, C, H, W = x["obs"]["image"].size()
            k = min(n_examples, B)

            actions = actions[:k]
            x = dict_apply(x, lambda x: x[:k])

            if cfg.task.dataset.language_emb_model is not None:
                if "language" in x["obs"]:
                    language_goal = x["obs"]["language"]
                    del x["obs"]["language"]
                elif "language_latents" in x:
                    language_goal = x["language_latents"]
                    del x["language_latents"]
                else:
                    raise NotImplementedError
            else:
                language_goal = None

            (
                x,
                real,
                _,
                c,
                text_latents,
                history_trajectory,
                trajectory,
                proprioception_input,
            ) = prepare_data_predict_action(
                cfg, x, actions, model, T, device, language_goal=language_goal
            )

            z, act_out = model.model.sample_tokens( # this is MAR (masked autoregressive model) from mar_con_unified.py
                bsz=k,
                cond=c,
                text_latents=text_latents,
                num_iter=cfg.model.policy.autoregressive_model_params.num_iter,
                cfg=cfg.model.policy.autoregressive_model_params.cfg,
                cfg_schedule=cfg.model.policy.autoregressive_model_params.cfg_schedule,
                temperature=cfg.model.policy.autoregressive_model_params.temperature,
                history_nactions=history_trajectory,
                nactions=trajectory,
                proprioception_input=proprioception_input,
                task_mode="full_dynamic_model",
            )
            pred = decode_from_sample_autoregressive(model.vae_model, z / 0.2325)
            pred = pred.clamp(-1, 1).cpu()

            pred = 1 + rearrange(pred, "(b t) c h w -> b t h w c", b=k)
            real = (1 + rearrange(real, "b c t h w -> b t h w c")).cpu()

            pred = pred * 127.5
            pred = pred.type(torch.uint8)

            real = real * 127.5
            real = real.type(torch.uint8)

            x = (1 + x) * 127.5  # b c t h w
            x = x.type(torch.uint8).cpu()

            if len(predictions) < n_examples:
                reals.append(
                    torch.cat(
                        [
                            x[:, :, : x.size(2) // 2],
                            rearrange(real, "b t h w c -> b c t h w"),
                        ],
                        dim=2,
                    )
                )
                predictions.append(
                    torch.cat(
                        [
                            x[:, :, : x.size(2) // 2],
                            rearrange(pred, "b t h w c -> b c t h w"),
                        ],
                        dim=2,
                    )
                )

            if real.shape[1] < 16:
                pred = pred.repeat_interleave(repeats=4, dim=1)
                real = real.repeat_interleave(repeats=4, dim=1)

            pred_embeddings.append(get_fvd_logits(pred.numpy(), i3d=i3d, device=device))
            real_embeddings.append(get_fvd_logits(real.numpy(), i3d=i3d, device=device))

    log_data = dict()
    reals = torch.cat(reals)
    predictions = torch.cat(predictions)

    real_embeddings = torch.cat(real_embeddings)
    pred_embeddings = torch.cat(pred_embeddings)
    fvd = frechet_distance(
        pred_embeddings.clone().detach(), real_embeddings.clone().detach()
    )
    fvd = fvd.item()

    os.makedirs(output_dir + "/vis", exist_ok=True)
    real_vid = save_image_grid(
        reals.cpu().numpy(),
        os.path.join(output_dir, f"vis/{name_label}real_{it}.gif"),
        drange=[0, 255],
        grid_size=(reals.size(0) // 4, 4),
    )  # [4, 3, 8, 128, 128]
    pred_vid = save_image_grid(
        predictions.cpu().numpy(),
        os.path.join(output_dir, f"vis/{name_label}predicted_{it}.gif"),
        drange=[0, 255],
        grid_size=(predictions.size(0) // 4, 4),
    )  # [4, 3, 8, 128, 128]

    real_video = wandb.Video(os.path.join(output_dir, f"vis/{name_label}real_{it}.mp4"))
    pred_video = wandb.Video(
        os.path.join(output_dir, f"vis/{name_label}predicted_{it}.mp4")
    )

    log_data[f"{name_label}video_fvd"] = fvd
    log_data[f"{name_label}real_img"] = real_video
    log_data[f"{name_label}predicted_img"] = pred_video

    return log_data


def test_video_fvd_extended(
    cfg, model, loader, it, output_dir, device, 
    n_cond_frames=4, 
    n_pred_frames=12,
    name_label="extended_", 
    plot_actions=False
):
    """
    Extended video generation using autoregressive rollout.
    
    The model natively generates 4 frames from 4 conditioning frames.
    This function chains multiple generation steps to produce longer videos.
    
    Args:
        cfg: Config from uva_nymeria.yaml
        model: UnifiedVideoActionPolicy
        loader: Validation dataloader
        it: Epoch index for logging
        output_dir: Output directory for saving videos
        device: CUDA device
        n_cond_frames: Number of initial conditioning frames (default: 4)
        n_pred_frames: Total number of frames to generate (default: 12)
                       Must be a multiple of 4.
        name_label: Prefix for logged metrics
        plot_actions: Whether to plot actions (not used for video-only)
    
    Returns:
        log_data: Dictionary with FVD score and video paths
    
    Example usage in training:
        fvd_log = test_video_fvd_extended(
            cfg, policy, val_dataloader, epoch, output_dir, device,
            n_cond_frames=4, n_pred_frames=12  # 4 cond → 12 gen frames
        )
    """
    assert n_pred_frames % 4 == 0, "n_pred_frames must be a multiple of 4"
    n_rollouts = n_pred_frames // 4  # Number of autoregressive steps
    
    losses = dict()
    losses["fvd"] = AverageMeter()

    i3d = load_i3d_pretrained(device)
    real_embeddings = []
    pred_embeddings = []

    reals = []
    predictions = []

    n_examples = 4

    with torch.no_grad():
        for n, batch in enumerate(loader):
            if n % 10 == 0:
                print(f"test_video_fvd_extended (rollouts={n_rollouts})", n, len(loader))

            x = batch
            if n >= n_examples:
                break

            x = dict_apply(x, lambda x: x.to(device, non_blocking=True))
            actions = x["action"]

            if cfg.model.policy.use_history_action:
                x = dict_apply(x, lambda x: x[:, 1:])

            x = resize_image(cfg, x)

            B, T_total, C, H, W = x["obs"]["image"].size()
            k = min(n_examples, B)

            actions = actions[:k]
            x = dict_apply(x, lambda x: x[:k])

            # Get language embeddings if needed
            if cfg.task.dataset.language_emb_model is not None:
                if "language" in x["obs"]:
                    language_goal = x["obs"]["language"]
                    del x["obs"]["language"]
                elif "language_latents" in x:
                    language_goal = x["language_latents"]
                    del x["language_latents"]
                else:
                    raise NotImplementedError
            else:
                language_goal = None

            # Normalize
            nactions = normalize_action(
                normalizer=model.normalizer,
                normalizer_type=model.normalizer_type,
                actions=actions,
            )
            x_normed = normalize_obs(
                normalizer=model.normalizer, 
                normalizer_type=model.normalizer_type, 
                batch=x
            )

            # Get images and normalize to [-1, 1]
            images = x_normed["obs"]["image"]  # (B, T, C, H, W) in [0, 1]
            images = images * 2 - 1  # Convert to [-1, 1]
            images = rearrange(images, "b t c h w -> b c t h w")  # (B, C, T, H, W)
            
            # Initialize: take first n_cond_frames as conditioning
            cond_frames = images[:, :, :n_cond_frames]  # (B, C, 4, H, W)
            
            # Extract conditioning latents
            cond_latents, _ = extract_latent_autoregressive(model.vae_model, cond_frames)
            # cond_latents shape: (B, 4, C_latent, H_latent, W_latent)
            
            all_generated_frames = []
            current_cond = cond_latents  # Start with initial conditioning
            
            # Autoregressive rollout: generate n_rollouts * 4 frames
            for rollout_idx in range(n_rollouts):
                # Prepare conditioning: rearrange to (B, T, C, H, W) format for sample_tokens
                c = rearrange(current_cond, "b t c h w -> b t c h w")
                
                # Get trajectory info
                history_trajectory, trajectory = get_trajectory(
                    nactions, T_total,
                    cfg.model.policy.shift_action,
                    use_history_action=cfg.model.policy.use_history_action,
                )
                
                # Handle text latents
                text_latents = None
                if cfg.task.dataset.language_emb_model is not None:
                    if "umi" in cfg.task.name:
                        text_latents = language_goal
                    elif "libero" in cfg.task.name:
                        if cfg.task.dataset.language_emb_model == "clip":
                            text_tokens = {
                                "input_ids": language_goal[:, 0].long()[:, 0],
                                "attention_mask": language_goal[:, 0].long()[:, 1],
                            }
                            text_latents = extract_text_features(
                                model.text_model,
                                text_tokens,
                                language_emb_model=cfg.task.dataset.language_emb_model,
                            )
                
                # Generate 4 frames
                z_gen, _ = model.model.sample_tokens(
                    bsz=k,
                    cond=c,
                    text_latents=text_latents,
                    num_iter=cfg.model.policy.autoregressive_model_params.num_iter,
                    cfg=cfg.model.policy.autoregressive_model_params.cfg,
                    cfg_schedule=cfg.model.policy.autoregressive_model_params.cfg_schedule,
                    temperature=cfg.model.policy.autoregressive_model_params.temperature,
                    history_nactions=history_trajectory,
                    nactions=trajectory,
                    proprioception_input={},
                    task_mode="full_dynamic_model",
                )
                
                # Decode generated latents to images
                pred_images = decode_from_sample_autoregressive(model.vae_model, z_gen / 0.2325)
                pred_images = pred_images.clamp(-1, 1)  # (B*4, C, H, W)
                pred_images = rearrange(pred_images, "(b t) c h w -> b t c h w", b=k)  # (B, 4, C, H, W)
                
                all_generated_frames.append(pred_images)
                
                # Use generated frames as new conditioning for next rollout
                # Re-encode to latent space
                pred_for_cond = rearrange(pred_images, "b t c h w -> b c t h w")
                current_cond, _ = extract_latent_autoregressive(model.vae_model, pred_for_cond)
            
            # Concatenate all generated frames: (B, n_pred_frames, C, H, W)
            all_generated = torch.cat(all_generated_frames, dim=1)
            
            # Get ground truth frames for comparison
            # Real frames: starting from n_cond_frames
            total_needed = n_cond_frames + n_pred_frames
            if T_total >= total_needed:
                real_frames = images[:, :, n_cond_frames:total_needed]  # (B, C, n_pred_frames, H, W)
            else:
                # Pad with last frame if not enough
                real_frames = images[:, :, n_cond_frames:]
                pad_size = n_pred_frames - real_frames.size(2)
                if pad_size > 0:
                    real_frames = torch.cat([
                        real_frames, 
                        real_frames[:, :, -1:].repeat(1, 1, pad_size, 1, 1)
                    ], dim=2)
            
            # Convert to display format
            pred = 1 + rearrange(all_generated, "b t c h w -> b t h w c")  # [0, 2]
            real = (1 + rearrange(real_frames, "b c t h w -> b t h w c")).cpu()
            
            pred = pred * 127.5
            pred = pred.type(torch.uint8).cpu()
            
            real = real * 127.5
            real = real.type(torch.uint8)
            
            # Get conditioning frames for visualization
            cond_vis = (1 + rearrange(cond_frames, "b c t h w -> b t h w c")).cpu()
            cond_vis = (cond_vis * 127.5).type(torch.uint8)
            
            # Convert for visualization: concat conditioning + prediction
            x_display = (1 + images) * 127.5
            x_display = x_display.type(torch.uint8).cpu()
            
            if len(predictions) < n_examples:
                # For visualization: [cond (4 frames) | pred (n_pred_frames)]
                cond_for_vis = rearrange(cond_vis, "b t h w c -> b c t h w")
                pred_for_vis = rearrange(pred, "b t h w c -> b c t h w")
                real_for_vis = rearrange(real, "b t h w c -> b c t h w")
                
                reals.append(torch.cat([cond_for_vis, real_for_vis], dim=2))
                predictions.append(torch.cat([cond_for_vis, pred_for_vis], dim=2))

            # FVD calculation needs at least 16 frames, repeat if needed
            if pred.shape[1] < 16:
                repeat_factor = (16 + pred.shape[1] - 1) // pred.shape[1]
                pred_fvd = pred.repeat_interleave(repeats=repeat_factor, dim=1)[:, :16]
                real_fvd = real.repeat_interleave(repeats=repeat_factor, dim=1)[:, :16]
            else:
                pred_fvd = pred[:, :16]
                real_fvd = real[:, :16]

            pred_embeddings.append(get_fvd_logits(pred_fvd.numpy(), i3d=i3d, device=device))
            real_embeddings.append(get_fvd_logits(real_fvd.numpy(), i3d=i3d, device=device))

    log_data = dict()
    reals = torch.cat(reals)
    predictions = torch.cat(predictions)

    real_embeddings = torch.cat(real_embeddings)
    pred_embeddings = torch.cat(pred_embeddings)
    fvd = frechet_distance(
        pred_embeddings.clone().detach(), real_embeddings.clone().detach()
    )
    fvd = fvd.item()

    os.makedirs(output_dir + "/vis", exist_ok=True)
    real_vid = save_image_grid(
        reals.cpu().numpy(),
        os.path.join(output_dir, f"vis/{name_label}real_{it}.gif"),
        drange=[0, 255],
        grid_size=(reals.size(0) // 4, 4),
    )
    pred_vid = save_image_grid(
        predictions.cpu().numpy(),
        os.path.join(output_dir, f"vis/{name_label}predicted_{it}.gif"),
        grid_size=(predictions.size(0) // 4, 4),
    )

    real_video = wandb.Video(os.path.join(output_dir, f"vis/{name_label}real_{it}.mp4"))
    pred_video = wandb.Video(
        os.path.join(output_dir, f"vis/{name_label}predicted_{it}.mp4")
    )

    log_data[f"{name_label}video_fvd"] = fvd
    log_data[f"{name_label}real_img"] = real_video
    log_data[f"{name_label}predicted_img"] = pred_video
    log_data[f"{name_label}n_cond_frames"] = n_cond_frames
    log_data[f"{name_label}n_pred_frames"] = n_pred_frames

    print(f"Extended video FVD ({n_cond_frames} cond → {n_pred_frames} pred): {fvd:.2f}")

    return log_data


def test_action_l2(
    cfg,
    model,
    loader,
    it,
    output_dir,
    device,
    text_model=None,
    name_label="",
    plot_actions=False,
):
    action_l2_distances = []

    with torch.no_grad():
        for n, batch in enumerate(loader):
            if n % 10 == 0:
                print("test_action_l2", n, len(loader))

            x = batch
            x = dict_apply(x, lambda x: x.to(device, non_blocking=True))
            actions = x["action"]

            if cfg.model.policy.use_history_action:
                x = dict_apply(x, lambda x: x[:, 1:])

            x = resize_image(cfg, x)

            B, T, C, H, W = x["obs"]["image"].size()

            if cfg.task.dataset.language_emb_model is not None:
                if "language" in x["obs"]:
                    language_goal = x["obs"]["language"]
                    del x["obs"]["language"]
                elif "language_latents" in x:
                    language_goal = x["language_latents"]
                    del x["language_latents"]
                else:
                    raise NotImplementedError
            else:
                language_goal = None

            (
                x,
                real,
                _,
                c,
                text_latents,
                history_trajectory,
                trajectory,
                proprioception_input,
            ) = prepare_data_predict_action(
                cfg, x, actions, model, T, device, language_goal=language_goal
            )

            z, act_out = model.model.sample_tokens(
                bsz=B,
                cond=c,
                text_latents=text_latents,
                num_iter=cfg.model.policy.autoregressive_model_params.num_iter,
                cfg=cfg.model.policy.autoregressive_model_params.cfg,
                cfg_schedule=cfg.model.policy.autoregressive_model_params.cfg_schedule,
                temperature=cfg.model.policy.autoregressive_model_params.temperature,
                history_nactions=history_trajectory,
                nactions=trajectory,
                proprioception_input=proprioception_input,
                task_mode="policy_model",
            )

            if cfg.model.policy.action_model_params.predict_action:
                act_out = unnormalize_future_action(
                    normalizer=model.normalizer,
                    normalizer_type=model.normalizer_type,
                    actions=act_out,
                )
                trajectory = unnormalize_future_action(
                    normalizer=model.normalizer,
                    normalizer_type=model.normalizer_type,
                    actions=trajectory,
                )

                ## calculate l2 distance between the predicted action and ground truth action
                l2_distance = torch.sqrt(
                    torch.sum((trajectory[:, :, :9] - act_out[:, :, :9]) ** 2, dim=-1)
                )
                action_l2_distances.append(l2_distance.mean())

            if cfg.training.debug:
                break

    log_data = dict()
    if cfg.model.policy.action_model_params.predict_action:
        log_data[f"{name_label}val_action_l2_distances"] = (
            torch.stack(action_l2_distances).mean().item()
        )

    return log_data
