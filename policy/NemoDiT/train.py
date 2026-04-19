# Training Script for Qwen3VL + Flow Matching Policy
#
# Architecture:
#   Qwen3-VL (frozen or LoRA) -> VLM hidden states
#   -> VLM projection -> DiT context
#   -> Cross-Attention DiT (flow matching) -> action predictions
#
# Training strategy:
#   - VLM: frozen (default) or LoRA fine-tuned
#   - VLM projection + DiT: fully trained
#   - Flow matching loss: MSE on velocity prediction
#
# Reference: ABot-Manipulation training pipeline

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
import argparse
import math
from pathlib import Path

from model.action_model.action_model import Qwen3VLActionModel
from dataloader import Qwen3VLRobotDataset, collate_fn
from utils.wandb_utils import WandbLogger
from utils.ema_model import EMAModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train Qwen3VL + Flow Matching for robot action generation')

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to robot dataset directory containing .hdf5 files')
    parser.add_argument('--num_cameras', type=int, default=4)
    parser.add_argument('--use_both_arms', action='store_true', default=True)
    parser.add_argument('--action_type', type=str, default='joint',
                        choices=['endpose', 'joint'])
    parser.add_argument('--quat_convention', type=str, default='wxyz',
                        choices=['wxyz', 'xyzw'])
    parser.add_argument('--instruction', type=str, default='Predict the next robot actions.',
                        help='Fallback task instruction used when no per-episode '
                             'JSON instruction file is available.')
    parser.add_argument('--instructions_path', type=str, default=None,
                        help='Directory containing per-episode instruction JSON files '
                             '(episode{N}.json with "seen"/"unseen" lists). '
                             'Defaults (None or empty) to <data_path>/../instructions. '
                             'Pass "disable" to turn off auto-loading and use --instruction.')
    parser.add_argument('--instruction_split', type=str, default='seen',
                        choices=['seen', 'unseen'],
                        help='Which split of the instruction JSON to sample from.')
    parser.add_argument('--no_random_instruction', action='store_true', default=False,
                        help='Always use the first instruction in the JSON list '
                             'instead of randomly sampling one per step.')

    # VLM arguments
    parser.add_argument('--vlm_model_name', type=str, default='Qwen/Qwen3-VL-4B-Instruct',
                        help='Qwen3-VL model name or local path')
    parser.add_argument('--freeze_vlm', action='store_true', default=True,
                        help='Freeze VLM weights (default: True)')
    parser.add_argument('--no_freeze_vlm', action='store_false', dest='freeze_vlm',
                        help='Unfreeze VLM weights for full fine-tuning')
    parser.add_argument('--use_lora', action='store_true', default=False,
                        help='Apply LoRA to VLM')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')

    # DiT model arguments
    parser.add_argument('--dit_model_type', type=str, default='DiT-B',
                        choices=['DiT-S', 'DiT-B', 'DiT-L', 'DiT-XL'],
                        help='DiT model size for action head')
    parser.add_argument('--action_dim', type=int, default=14,
                        help='Action dimension (auto-set from action_type + use_both_arms)')
    parser.add_argument('--future_action_window', type=int, default=13,
                        help='Number of future action steps to predict')
    parser.add_argument('--n_obs_steps', type=int, default=1,
                        help='Number of observation history steps')
    parser.add_argument('--n_action_steps', type=int, default=8,
                        help='Number of action steps to execute during inference')

    # Flow Matching arguments
    parser.add_argument('--time_sampling', type=str, default='logit_normal',
                        choices=['logit_normal', 'beta', 'uniform'])
    parser.add_argument('--logit_normal_loc', type=float, default=0.0)
    parser.add_argument('--logit_normal_scale', type=float, default=1.0)
    parser.add_argument('--beta_alpha', type=float, default=1.5)
    parser.add_argument('--beta_beta', type=float, default=1.0)
    parser.add_argument('--num_timestep_buckets', type=int, default=1000)
    parser.add_argument('--num_inference_steps', type=int, default=10)

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per GPU (smaller due to VLM memory)')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--vlm_lr', type=float, default=1e-5,
                        help='Learning rate for VLM parameters (when unfrozen or LoRA)')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Use Automatic Mixed Precision training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps for effective larger batch')

    # Warmup
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--warmup_type', type=str, default='linear',
                        choices=['linear', 'cosine'])

    # EMA
    parser.add_argument('--use_ema', action='store_true', default=False)
    parser.add_argument('--ema_inv_gamma', type=float, default=1.0)
    parser.add_argument('--ema_power', type=float, default=0.6667)
    parser.add_argument('--ema_max_value', type=float, default=0.9999)

    # VLM caching
    parser.add_argument('--cache_vlm_features', action='store_true', default=False,
                        help='Pre-compute and cache VLM features (saves memory during training)')

    # Checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')

    # WandB
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='robotwin_qwen3vl_fm')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_name', type=str, default=None)

    return parser.parse_args()


def prepare_dataloader(args):
    """Prepare dataset and dataloader."""
    dataset = Qwen3VLRobotDataset(
        data_path=args.data_path,
        future_action_window=args.future_action_window,
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

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    return dataloader, dataset


def create_model(args):
    """Create Qwen3VL + Flow Matching action model."""
    model = Qwen3VLActionModel(
        vlm_model_name=args.vlm_model_name,
        freeze_vlm=args.freeze_vlm,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        dit_model_type=args.dit_model_type,
        action_dim=args.action_dim,
        future_action_window_size=args.future_action_window,
        n_action_steps=args.n_action_steps,
        time_sampling=args.time_sampling,
        logit_normal_loc=args.logit_normal_loc,
        logit_normal_scale=args.logit_normal_scale,
        beta_alpha=args.beta_alpha,
        beta_beta=args.beta_beta,
        num_timestep_buckets=args.num_timestep_buckets,
    )
    return model


def create_optimizer(model, args):
    """
    Create optimizer with separate learning rates for VLM and action head.

    VLM parameters (if trainable) use a lower learning rate.
    Action head (vlm_proj + DiT + flow matching) uses the main learning rate.
    """
    vlm_params = []
    action_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('vlm.'):
            vlm_params.append(param)
        else:
            action_params.append(param)

    param_groups = [
        {'params': action_params, 'lr': args.lr},
    ]

    if vlm_params:
        param_groups.append({'params': vlm_params, 'lr': args.vlm_lr})
        print(f"VLM trainable params: {sum(p.numel() for p in vlm_params):,} (lr={args.vlm_lr})")

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    return optimizer


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, warmup_type='linear'):
    """Create warmup + cosine annealing scheduler."""
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            if warmup_type == 'linear':
                return (current_epoch + 1) / warmup_epochs
            else:
                return 0.5 * (1 - math.cos(math.pi * (current_epoch + 1) / warmup_epochs))
        else:
            progress = (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, args,
                    filename=None, ema_model=None):
    """Save training checkpoint."""
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if filename is None:
        filename = f'{epoch}.pt'

    checkpoint_path = os.path.join(args.checkpoint_dir, filename)

    checkpoint = {
        'model_state_dict': model.state_dict(),
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

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    latest_path = os.path.join(args.checkpoint_dir, 'latest.pt')
    torch.save(checkpoint, latest_path)


def load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path, ema_model=None):
    """Load training checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    if ema_model is not None and 'ema_state_dict' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_state_dict'])

    epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']
    print(f"Resumed from epoch {epoch}, global step {global_step}")

    return epoch, global_step


def train_step(model, batch, device, args):
    """
    Single training step with VLM encoding + flow matching loss.

    For each batch:
    1. Encode images + instruction through Qwen3-VL -> context
    2. Compute flow matching loss using context as conditioning
    """
    actions = batch['actions'].to(device)   # (B, T-1, action_dim)
    state = batch['state'].to(device)       # (B, action_dim)

    # Encode through VLM: process each sample's images
    # batch['images_raw'] is List[List[np.ndarray]] - [B][num_cameras]
    images_raw_batch = batch['images_raw']
    instructions = batch['instruction']

    # Process VLM for the batch
    # For efficiency, we process VLM per-sample and stack contexts
    contexts = []
    for i in range(len(images_raw_batch)):
        context = model.encode_vlm(
            images=images_raw_batch[i],
            instruction=instructions[i],
            robot_state=state[i],
        )
        contexts.append(context)

    # Stack contexts: all should have same seq length (padded by processor)
    # Find max length and pad
    max_len = max(c.shape[1] for c in contexts)
    padded_contexts = []
    for c in contexts:
        if c.shape[1] < max_len:
            pad = torch.zeros(1, max_len - c.shape[1], c.shape[2],
                            device=c.device, dtype=c.dtype)
            c = torch.cat([c, pad], dim=1)
        padded_contexts.append(c)

    context = torch.cat(padded_contexts, dim=0)  # (B, L, dit_hidden_size)

    # Compute flow matching loss
    loss = model.loss(actions=actions, context=context, state=state)

    return loss


def train():
    """Main training loop."""
    args = parse_args()

    # Auto-set action_dim
    if args.action_type == 'endpose':
        args.action_dim = 20 if args.use_both_arms else 10
    else:
        args.action_dim = 14 if args.use_both_arms else 7

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"VLM: {args.vlm_model_name} (frozen={args.freeze_vlm}, lora={args.use_lora})")
    print(f"DiT: {args.dit_model_type}, action_dim={args.action_dim}")
    print(f"Action type: {args.action_type} ({'dual arm' if args.use_both_arms else 'single arm'})")

    # Initialize WandB
    wandb_logger = None
    if args.use_wandb:
        run_name = args.wandb_name or f"qwen3vl_{args.dit_model_type}_{args.action_type}"
        wandb_logger = WandbLogger(
            project_name=args.wandb_project,
            run_name=run_name,
            config=vars(args),
            entity=args.wandb_entity,
        )

    # Create dataloader
    print("Loading dataset...")
    dataloader, dataset = prepare_dataloader(args)
    print(f"Dataset size: {len(dataset)}, Batches: {len(dataloader)}")

    # Create model
    print("Creating model...")
    model = create_model(args)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    # Create optimizer with separate LR groups
    optimizer = create_optimizer(model, args)

    # Scheduler
    if args.warmup_epochs > 0:
        scheduler = get_warmup_cosine_scheduler(
            optimizer, args.warmup_epochs, args.epochs, args.warmup_type)
        print(f"Using warmup: {args.warmup_epochs} epochs ({args.warmup_type})")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # AMP
    scaler = GradScaler() if args.use_amp else None
    if args.use_amp:
        print("Using Automatic Mixed Precision (AMP)")

    # EMA
    ema_model = None
    if args.use_ema:
        ema_model = EMAModel(
            model, inv_gamma=args.ema_inv_gamma,
            power=args.ema_power, max_value=args.ema_max_value)
        print(f"Using EMA (max_value={args.ema_max_value})")

    # Resume
    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(
            model, optimizer, scheduler, scaler, args.resume, ema_model)

    # Training loop
    print("Starting training...")
    model.train()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, batch in enumerate(progress_bar):
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
                ema_model.step(model)

            epoch_loss += loss.item() * args.gradient_accumulation_steps
            global_step += 1

            if wandb_logger:
                wandb_logger.log({
                    'train/loss': loss.item() * args.gradient_accumulation_steps,
                    'train/lr': optimizer.param_groups[0]["lr"],
                    'train/epoch': epoch + 1,
                    'train/global_step': global_step,
                }, step=global_step)

            progress_bar.set_postfix({
                'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}',
                'avg': f'{epoch_loss / (batch_idx + 1):.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}',
            })

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_loss:.4f}")

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, scaler,
                          epoch + 1, global_step, args, ema_model=ema_model)

    print("Training completed!")
    save_checkpoint(model, optimizer, scheduler, scaler,
                  args.epochs, global_step, args, filename='final.pt', ema_model=ema_model)

    if wandb_logger:
        wandb_logger.finish()


if __name__ == '__main__':
    train()
