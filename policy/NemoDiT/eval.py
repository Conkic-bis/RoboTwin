# Offline eval for the Qwen3-VL + Flow Matching ActionModel.
#
# Two API changes vs the 2.0 branch:
#   - `model.sample()` now accepts `images` (raw per-camera uint8 arrays)
#     + `instructions` + `state`, not an (n_obs, C, H, W) tensor.
#   - `RobotDataset.__getitem__` returns `images_raw` (last obs frame,
#     list of per-camera uint8 images) instead of `images`. We use that
#     directly so the VLM processor handles normalization.

import argparse

import numpy as np
import torch

from dataloader import RobotDataset
from model.action_model.action_model import ActionModel


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate Qwen3VL + Flow Matching ActionModel')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--data_path', type=str, default=None)
    p.add_argument('--num_samples', type=int, default=5)
    p.add_argument('--num_inference_steps', type=int, default=10)
    p.add_argument('--ode_solver', type=str, default='midpoint', choices=['midpoint', 'euler'])
    p.add_argument('--cfg_scale', type=float, default=1.0)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def load_model(ckpt_path: str, device: torch.device):
    print(f"[eval] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    tr = ckpt['args']
    print(f"[eval] model_type={tr['model_type']}  action_dim={tr['action_dim']}  "
          f"n_obs_steps={tr['n_obs_steps']}  n_action_steps={tr['n_action_steps']}  "
          f"future_action_window={tr['future_action_window']}")

    model = ActionModel(
        model_type=tr['model_type'],
        in_channels=tr['action_dim'],
        future_action_window_size=tr['future_action_window'],
        past_action_window_size=tr['past_action_window'],
        n_obs_steps=tr['n_obs_steps'],
        n_action_steps=tr['n_action_steps'],
        vlm_model_name=tr.get('vlm_model_name', 'Qwen/Qwen3-VL-4B-Instruct'),
        freeze_vlm=tr.get('freeze_vlm', True),
        use_lora=tr.get('use_lora', False),
        lora_r=tr.get('lora_r', 16),
        lora_alpha=tr.get('lora_alpha', 32),
        time_sampling=tr.get('time_sampling', 'logit_normal'),
        logit_normal_loc=tr.get('logit_normal_loc', 0.0),
        logit_normal_scale=tr.get('logit_normal_scale', 1.0),
        beta_alpha=tr.get('beta_alpha', 1.5),
        beta_beta=tr.get('beta_beta', 1.0),
        num_timestep_buckets=tr.get('num_timestep_buckets', 1000),
        # legacy (ignored by ctor)
        token_size=tr.get('token_size', 2048),
        vision_backbone_type=tr.get('vision_backbone', None),
        vision_pretrained=False,
        num_cameras=tr.get('num_cameras', 4),
        adapter_type=tr.get('adapter_type', None),
        class_dropout_prob=0.0,
        temporal_agg=tr.get('temporal_agg', 'last'),
    )

    # Checkpoints from train.save_checkpoint are always unwrapped.
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    missing_non_vlm = [k for k in missing if not k.startswith('vlm.')]
    if missing_non_vlm:
        print(f"[eval] WARN missing non-vlm keys: {len(missing_non_vlm)}")
    if unexpected:
        print(f"[eval] WARN unexpected keys: {len(unexpected)}")

    model = model.to(device).eval()
    print(f"[eval] loaded epoch={ckpt.get('epoch', 'N/A')}")
    return model, tr


def prepare_eval_dataset(data_path: str, tr: dict) -> RobotDataset:
    return RobotDataset(
        data_path=data_path,
        future_action_window=tr['future_action_window'],
        past_action_window=tr['past_action_window'],
        transform=None,                  # VLM handles its own preproc
        num_cameras=tr['num_cameras'],
        use_both_arms=tr.get('use_both_arms', False),
        action_type=tr.get('action_type', 'endpose'),
        quat_convention=tr.get('quat_convention', 'wxyz'),
        n_obs_steps=tr['n_obs_steps'],
        instruction=tr.get('instruction', 'Predict the next robot actions.'),
        instructions_path=tr.get('instructions_path', None),
        instruction_split=tr.get('instruction_split', 'seen'),
        random_instruction=False,         # deterministic during eval
    )


@torch.no_grad()
def evaluate_sample(model, images_raw, instruction, state, gt_actions,
                    num_steps, ode_solver, cfg_scale, device):
    state = state.to(device) if state is not None else None
    gt = gt_actions.to(device)

    pred = model.sample(
        images=[images_raw],            # B=1: wrap the per-camera list
        instructions=[instruction],
        state=state,
        num_steps=num_steps,
        ode_solver=ode_solver,
        cfg_scale=cfg_scale,
        return_all=False,
    )  # (1, n_action_steps, action_dim)

    T = pred.shape[1]
    gt_t = gt[:, :T, :]
    mse = ((pred - gt_t) ** 2).mean().item()
    l1 = (pred - gt_t).abs().mean().item()
    return pred.squeeze(0).cpu().numpy(), {'mse': mse, 'l1': l1}


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[eval] device={device}")

    model, tr = load_model(args.checkpoint, device)

    if args.data_path is None:
        print("\n=== API sketch ===")
        print(
            "actions = model.sample(\n"
            "    images=[[cam0, cam1, cam2, cam3]],         # uint8 HxWx3 list per sample\n"
            "    instructions=['pick up the block'],\n"
            "    state=state_tensor,        # (B, action_dim)\n"
            "    num_steps=10,\n"
            "    ode_solver='midpoint',\n"
            ")\n"
        )
        return

    ds = prepare_eval_dataset(args.data_path, tr)
    print(f"[eval] dataset={len(ds)}")

    n = min(args.num_samples, len(ds))
    mses, l1s = [], []
    for i in range(n):
        sample = ds[i]
        images_raw = sample['images_raw']                      # List[np.uint8]
        instruction = sample['instruction']
        state = sample['state'].unsqueeze(0)                   # (1, action_dim)
        gt_actions = sample['actions'].unsqueeze(0)            # (1, T-1, action_dim)

        pred, metrics = evaluate_sample(
            model, images_raw, instruction, state, gt_actions,
            args.num_inference_steps, args.ode_solver, args.cfg_scale, device,
        )
        mses.append(metrics['mse'])
        l1s.append(metrics['l1'])
        print(f"  sample {i + 1}: MSE={metrics['mse']:.6f}  L1={metrics['l1']:.6f}  "
              f"pred_shape={pred.shape}")

    print("-" * 60)
    print(f"mean MSE = {np.mean(mses):.6f} (+/- {np.std(mses):.6f})")
    print(f"mean L1  = {np.mean(l1s):.6f} (+/- {np.std(l1s):.6f})")


if __name__ == '__main__':
    main()
