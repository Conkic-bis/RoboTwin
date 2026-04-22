# Training script for Qwen3-VL + Flow Matching action policy.
#
# Key behaviours (different from the 2.0 branch):
#   - VLM forward is batched once per step (not looped per sample), matching
#     the DiT batch dimension exactly. Attention mask is threaded into the
#     cross-attention so that padded VLM tokens contribute zero softmax
#     weight.
#   - DiT and vlm_proj are torch.compiled by default. EMA is constructed
#     BEFORE compile so shadow parameters reference the unwrapped module.
#     state_dict saved/loaded always uses the unwrapped module so that
#     checkpoints are portable between compiled and eager runs.
#   - Dataloader uses persistent_workers=True so per-worker HDF5 handles
#     are opened once per training run, not once per epoch.

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader import RobotDataset, collate_fn
from model.action_model.action_model import ActionModel
from utils.ema_model import EMAModel
from utils.wandb_utils import WandbLogger


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Qwen3VL + Flow Matching training')

    # Data
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--num_cameras', type=int, default=4)
    p.add_argument('--use_both_arms', action='store_true', default=True)
    p.add_argument('--action_type', type=str, default='joint', choices=['endpose', 'joint'])
    p.add_argument('--quat_convention', type=str, default='wxyz', choices=['wxyz', 'xyzw'])
    p.add_argument('--instruction', type=str, default='Predict the next robot actions.')
    p.add_argument('--instructions_path', type=str, default=None)
    p.add_argument('--instruction_split', type=str, default='seen', choices=['seen', 'unseen'])
    p.add_argument('--no_random_instruction', action='store_true', default=False)

    # VLM
    p.add_argument('--vlm_model_name', type=str, default='Qwen/Qwen3-VL-4B-Instruct')
    p.add_argument('--freeze_vlm', action='store_true', default=True)
    p.add_argument('--no_freeze_vlm', action='store_false', dest='freeze_vlm')
    p.add_argument('--use_lora', action='store_true', default=False)
    p.add_argument('--lora_r', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)

    # DiT
    p.add_argument('--model_type', type=str, default='DiT-B',
                   choices=['DiT-S', 'DiT-B', 'DiT-L', 'DiT-XL'])
    p.add_argument('--action_dim', type=int, default=14)
    p.add_argument('--future_action_window', type=int, default=13)
    p.add_argument('--past_action_window', type=int, default=0)
    p.add_argument('--n_obs_steps', type=int, default=1)
    p.add_argument('--n_action_steps', type=int, default=8)
    p.add_argument('--token_size', type=int, default=2048,   # legacy, ignored
                   help='(Deprecated) ignored; retained for old shell scripts.')
    p.add_argument('--temporal_agg', type=str, default='last',
                   choices=['last', 'mean', 'concat'])
    p.add_argument('--dropout_prob', type=float, default=0.1)

    # Vision (legacy, ignored)
    p.add_argument('--vision_backbone', type=str, default=None)
    p.add_argument('--vision_pretrained', action='store_true', default=True)
    p.add_argument('--freeze_vision', action='store_true', default=False)
    p.add_argument('--adapter_type', type=str, default=None)
    p.add_argument('--no_resize', action='store_true', default=True)
    p.add_argument('--image_size', type=int, default=224)

    # Flow matching
    p.add_argument('--time_sampling', type=str, default='logit_normal',
                   choices=['logit_normal', 'beta', 'uniform'])
    p.add_argument('--logit_normal_loc', type=float, default=0.0)
    p.add_argument('--logit_normal_scale', type=float, default=1.0)
    p.add_argument('--beta_alpha', type=float, default=1.5)
    p.add_argument('--beta_beta', type=float, default=1.0)
    p.add_argument('--num_timestep_buckets', type=int, default=1000)
    p.add_argument('--num_inference_steps', type=int, default=10)

    # Legacy diffusion kwargs (ignored, kept for old shell scripts)
    p.add_argument('--diffusion_steps', type=int, default=None)
    p.add_argument('--noise_schedule', type=str, default=None)

    # Training
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--vlm_lr', type=float, default=1e-5)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--use_amp', action='store_true', default=False)
    p.add_argument('--gradient_accumulation_steps', type=int, default=1)

    # Warmup + scheduler
    p.add_argument('--warmup_epochs', type=int, default=0)
    p.add_argument('--warmup_type', type=str, default='linear', choices=['linear', 'cosine'])

    # EMA
    p.add_argument('--use_ema', action='store_true', default=False)
    p.add_argument('--ema_inv_gamma', type=float, default=1.0)
    p.add_argument('--ema_power', type=float, default=0.6667)
    p.add_argument('--ema_max_value', type=float, default=0.9999)

    # Compile
    p.add_argument('--use_compile', action='store_true', default=True,
                   help='torch.compile DiT + vlm_proj (default: True).')
    p.add_argument('--no_compile', action='store_false', dest='use_compile',
                   help='Disable torch.compile (useful for debugging).')
    p.add_argument('--compile_mode', type=str, default='default',
                   choices=['default', 'reduce-overhead', 'max-autotune'])

    # Checkpoint
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    p.add_argument('--save_every', type=int, default=50)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--device', type=str, default='cuda')

    # WandB
    p.add_argument('--use_wandb', action='store_true', default=False)
    p.add_argument('--wandb_project', type=str, default='robotwin_qwen3vl_fm')
    p.add_argument('--wandb_entity', type=str, default=None)
    p.add_argument('--wandb_name', type=str, default=None)

    return p.parse_args()


# =============================================================================
# Build pipeline pieces
# =============================================================================

def prepare_dataloader(args):
    # Qwen3-VL has its own image normalization — the tensor path below is
    # only used by fallback code. Keep it minimal.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    dataset = RobotDataset(
        data_path=args.data_path,
        future_action_window=args.future_action_window,
        past_action_window=args.past_action_window,
        transform=transform,
        num_cameras=args.num_cameras,
        use_both_arms=args.use_both_arms,
        action_type=args.action_type,
        quat_convention=args.quat_convention,
        n_obs_steps=args.n_obs_steps,
        instruction=args.instruction,
        instructions_path=args.instructions_path,
        instruction_split=args.instruction_split,
        random_instruction=not args.no_random_instruction,
    )

    # persistent_workers=True makes the per-worker HDF5 handle cache (Fix A)
    # survive epoch boundaries instead of being torn down and rebuilt. When
    # num_workers=0 this kwarg is silently ignored by PyTorch.
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
    )
    return dataloader, dataset


def create_model(args):
    return ActionModel(
        model_type=args.model_type,
        in_channels=args.action_dim,
        future_action_window_size=args.future_action_window,
        past_action_window_size=args.past_action_window,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        vlm_model_name=args.vlm_model_name,
        freeze_vlm=args.freeze_vlm,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        time_sampling=args.time_sampling,
        logit_normal_loc=args.logit_normal_loc,
        logit_normal_scale=args.logit_normal_scale,
        beta_alpha=args.beta_alpha,
        beta_beta=args.beta_beta,
        num_timestep_buckets=args.num_timestep_buckets,
        # legacy kwargs (ignored internally but accepted for compat)
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        token_size=args.token_size,
        vision_backbone_type=args.vision_backbone,
        vision_pretrained=args.vision_pretrained,
        num_cameras=args.num_cameras,
        adapter_type=args.adapter_type,
        freeze_vision_backbone=args.freeze_vision,
        class_dropout_prob=args.dropout_prob,
        temporal_agg=args.temporal_agg,
    )


def create_optimizer(model, args):
    """Separate LR groups for VLM and action head."""
    vlm_params, action_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('vlm.'):
            vlm_params.append(param)
        else:
            action_params.append(param)

    param_groups = [{'params': action_params, 'lr': args.lr}]
    if vlm_params:
        param_groups.append({'params': vlm_params, 'lr': args.vlm_lr})
        print(f"[train] VLM trainable params: {sum(p.numel() for p in vlm_params):,}  "
              f"(lr={args.vlm_lr})")
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999),
                             weight_decay=args.weight_decay)


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, warmup_type='linear'):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            if warmup_type == 'linear':
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1 - math.cos(math.pi * (epoch + 1) / warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Checkpoint helpers — always save UNWRAPPED state_dict so ckpts are portable
# between compiled and non-compiled runs.
# =============================================================================

def _unwrap(mod):
    return getattr(mod, '_orig_mod', mod)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step,
                    args, filename=None, ema_model=None):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if filename is None:
        filename = f'{epoch}.pt'
    path = os.path.join(args.checkpoint_dir, filename)

    # Unwrap top-level submodules before flattening into a single state_dict
    # so key names match the eager module tree regardless of compile status.
    unwrapped_sd = {}
    for name, child in model.named_children():
        child_sd = _unwrap(child).state_dict()
        for k, v in child_sd.items():
            unwrapped_sd[f'{name}.{k}'] = v

    checkpoint = {
        'model_state_dict': unwrapped_sd,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'args': vars(args),
    }
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    if ema_model is not None:
        checkpoint['ema_state_dict'] = ema_model.state_dict()

    torch.save(checkpoint, path)
    torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'latest.pt'))
    print(f"[train] checkpoint saved: {path}")


def load_checkpoint(model, optimizer, scheduler, scaler, ckpt_path, ema_model=None):
    print(f"[train] loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Load into the unwrapped tree to avoid `_orig_mod.` key mismatches.
    # strict=False because VLM weights are typically skipped (frozen) and
    # may be absent from some checkpoints.
    for name, child in model.named_children():
        sub_prefix = name + '.'
        sub_sd = {k[len(sub_prefix):]: v
                  for k, v in ckpt['model_state_dict'].items()
                  if k.startswith(sub_prefix)}
        if sub_sd:
            _unwrap(child).load_state_dict(sub_sd, strict=False)

    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler is not None and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    if ema_model is not None and 'ema_state_dict' in ckpt:
        ema_model.load_state_dict(ckpt['ema_state_dict'])

    return ckpt['epoch'], ckpt['global_step']


# =============================================================================
# Training step
# =============================================================================

def train_step(model, batch, device, args):
    """One training step.

    Batching invariant (runtime-enforced): the VLM produces a context tensor
    of shape (B, L_max, H_dit) with a matching (B, L_max) mask; the DiT then
    consumes that same B against (B, T-1, action_dim) actions and
    (B, action_dim) state.
    """
    actions = batch['actions'].to(device, non_blocking=True)
    state = batch['state'].to(device, non_blocking=True)
    images_raw = batch['images_raw']         # List[List[np.ndarray]]
    instructions = batch['instruction']      # List[str]

    # Explicit cross-batch-size sanity check. This is the invariant the
    # prior implementation violated (per-sample VLM loop + ad-hoc padding).
    assert len(images_raw) == actions.shape[0] == state.shape[0], (
        f"batch dim mismatch: images={len(images_raw)}, "
        f"actions={actions.shape[0]}, state={state.shape[0]}"
    )

    # Single batched VLM forward (Fix C).
    # Calling through the compiled vlm_proj happens inside encode_vlm_batch;
    # vlm itself is deliberately NOT compiled (dynamic seq_len would cause
    # endless graph captures).
    context, context_mask = model.encode_vlm_batch(
        images_batch=images_raw,
        instructions=instructions,
        robot_states=state,
    )

    return model.loss(
        x=actions,
        context=context,
        context_mask=context_mask,
        state=state,
    )


# =============================================================================
# Main
# =============================================================================

def train():
    args = parse_args()

    # Auto-set action_dim when the user did not override it
    if args.action_type == 'endpose':
        args.action_dim = 20 if args.use_both_arms else 10
    else:
        args.action_dim = 14 if args.use_both_arms else 7

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[train] device={device}  VLM={args.vlm_model_name}  DiT={args.model_type}  "
          f"action_dim={args.action_dim}")

    wandb_logger = None
    if args.use_wandb:
        run_name = args.wandb_name or f"qwen3vl_{args.model_type}_{args.action_type}"
        wandb_logger = WandbLogger(
            project_name=args.wandb_project,
            run_name=run_name,
            config=vars(args),
            entity=args.wandb_entity,
        )

    # --- data ---
    print("[train] building dataset...")
    dataloader, dataset = prepare_dataloader(args)
    print(f"[train] dataset={len(dataset)}  batches/epoch={len(dataloader)}")

    # --- model ---
    print("[train] creating model...")
    model = create_model(args).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] params total={total:,}  trainable={trainable:,}")

    # --- optimizer + scheduler ---
    optimizer = create_optimizer(model, args)
    if args.warmup_epochs > 0:
        scheduler = get_warmup_cosine_scheduler(
            optimizer, args.warmup_epochs, args.epochs, args.warmup_type,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs,
        )

    scaler = GradScaler() if args.use_amp else None

    # --- EMA (BEFORE compile so shadow params reference the eager module) ---
    ema_model = None
    if args.use_ema:
        ema_model = EMAModel(
            model.dit,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            max_value=args.ema_max_value,
        )
        print(f"[train] EMA on DiT only, max_value={args.ema_max_value}")

    # --- torch.compile (AFTER .to(device) and AFTER EMA snapshot) ---
    if args.use_compile:
        # We do NOT compile model.vlm: the Qwen3-VL token length is
        # data-dependent (image grid varies), which causes repeated guard
        # failures and re-compilation. DiT has static shape (B, T-1, D)
        # and vlm_proj has static shape (B, L_max, H_vlm) for each step
        # — ideal compile targets. fullgraph=False on DiT keeps a small
        # escape hatch if timm's attention impl triggers a graph break.
        model.dit = torch.compile(model.dit, mode=args.compile_mode, fullgraph=False)
        model.vlm_proj = torch.compile(model.vlm_proj, mode=args.compile_mode, fullgraph=True)
        # Re-point the legacy `net` alias to the compiled DiT.
        model.net = model.dit
        print(f"[train] torch.compile enabled (mode={args.compile_mode})")
    else:
        print("[train] torch.compile DISABLED")

    # --- resume ---
    start_epoch, global_step = 0, 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(
            model, optimizer, scheduler, scaler, args.resume, ema_model,
        )

    # --- train ---
    print("[train] starting...")
    model.train()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"epoch {epoch + 1}/{args.epochs}")

        for batch_idx, batch in enumerate(pbar):
            if args.use_amp:
                with autocast():
                    loss = train_step(model, batch, device, args)
                loss = loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss = train_step(model, batch, device, args)
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

            if ema_model is not None:
                # EMA always tracks the unwrapped DiT; _unwrap handles the
                # compiled-module case.
                ema_model.step(_unwrap(model.dit))

            effective_loss = loss.item() * args.gradient_accumulation_steps
            epoch_loss += effective_loss
            global_step += 1

            if wandb_logger:
                wandb_logger.log({
                    'train/loss': effective_loss,
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'train/epoch': epoch + 1,
                    'train/global_step': global_step,
                }, step=global_step)

            pbar.set_postfix({
                'loss': f'{effective_loss:.4f}',
                'avg': f'{epoch_loss / (batch_idx + 1):.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}',
            })

        scheduler.step()
        print(f"[train] epoch {epoch + 1} avg_loss={epoch_loss / len(dataloader):.4f}")

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, scaler,
                            epoch + 1, global_step, args, ema_model=ema_model)

    save_checkpoint(model, optimizer, scheduler, scaler,
                    args.epochs, global_step, args,
                    filename='final.pt', ema_model=ema_model)
    print("[train] done.")

    if wandb_logger:
        wandb_logger.finish()


if __name__ == '__main__':
    train()
