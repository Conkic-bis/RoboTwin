import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from tqdm import tqdm
import os
import argparse
import math
from pathlib import Path

from model.action_model.action_model import ActionModel
from dataloader import RobotDataset
from utils.wandb_utils import WandbLogger
from utils.ema_model import EMAModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train DiT for robot action generation')

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to robot dataset directory containing .hdf5 files')
    parser.add_argument('--num_cameras', type=int, default=4,
                        help='Number of camera views (default: 4)')
    parser.add_argument('--use_both_arms', action='store_true', default=True,
                        help='Use both arms data (default: True)')
    parser.add_argument('--action_type', type=str, default='joint',
                        choices=['endpose', 'joint'],
                        help='Action type: endpose (ee pose) or joint (joint angles)')
    parser.add_argument('--quat_convention', type=str, default='wxyz',
                        choices=['wxyz', 'xyzw'],
                        help='Quaternion convention in HDF5 data (default: wxyz, only for endpose)')

    # Model arguments
    parser.add_argument('--model_type', type=str, default='DiT-S',
                        choices=['DiT-S', 'DiT-B', 'DiT-L', 'DiT-XL'],
                        help='DiT model size (default: DiT-B)')
    parser.add_argument('--dropout_prob', type=float, default=0.1,
                        help='Class dropout probability for classifier-free guidance (default: 0.1)')
    parser.add_argument('--action_dim', type=int, default=7,
                        help='Action dimension (default: 7 for 6-DoF joint angles + 1D gripper. 10 for 3D translation + 6D rot6d + 1D gripper)')
    parser.add_argument('--future_action_window', type=int, default=13,
                        help='Number of future action steps to predict (default: 13)')
    # Past_Action 会对模型Action造成扰动，仅预留接口
    parser.add_argument('--past_action_window', type=int, default=0,
                        help='Number of past action steps as context (default: 0)')
    # Token长度可根据输入Token情况更换
    parser.add_argument('--token_size', type=int, default=2048,
                        help='Token size for conditioning (default: 2048)')

    # Chunk机制参数
    # n_obs_steps + n_action_steps ＜ future_action_window
    # n_obs_steps: 观测步数，用于视觉编码的历史帧数
    # n_obs_steps 的最后一帧对应动作序列的第一帧时刻
    parser.add_argument('--n_obs_steps', type=int, default=3,
                        help='Number of observation steps for visual conditioning (default: 3)')
    # temporal_agg: 时间聚合方式(区别于Action_Chunk的时间聚合)
    # - 'last': 只使用最后一帧观测
    # - 'mean': 对所有观测帧取平均
    # - 'concat': 拼接所有帧特征后投影
    parser.add_argument('--temporal_agg', type=str, default='concat',
                        choices=['last', 'mean', 'concat'],
                        help='Temporal aggregation method for multi-frame observations (default: concat)')
    # n_action_steps: 动作执行步数，推理时实际执行的动作步数
    # 通常设置为 future_action_window 的一半或更少，用于 receding horizon control
    parser.add_argument('--n_action_steps', type=int, default=8,
                        help='Number of action steps to execute during inference (default: 8)')

    # Vision arguments
    parser.add_argument('--vision_backbone', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'vit_b_16', 'vit_b_32'],
                        help='Vision backbone type (default: resnet50)')
    parser.add_argument('--vision_pretrained', action='store_true', default=True,
                        help='Use pretrained vision backbone')
    parser.add_argument('--freeze_vision', action='store_true', default=False,
                        help='Freeze vision backbone weights')
    parser.add_argument('--adapter_type', type=str, default='mlp',
                        choices=['linear', 'mlp', 'attention_pooling'],
                        help='Feature adapter type')

    # Diffusion arguments
    parser.add_argument('--diffusion_steps', type=int, default=500,
                        help='Number of diffusion steps (default: 500)')
    parser.add_argument('--noise_schedule', type=str, default='squaredcos_cap_v2',
                        help='Noise schedule type (default: squaredcos_cap_v2)')

    # Training arguments 
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size per GPU (default: 16)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs (default: 500)')                      
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')


    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for L2 regularization (default: 0.01)')


    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm (default: 1.0)')


    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')

    # Mixed Precision Training (AMP)
    # 使用 FP16 混合精度训练，可显著减少显存占用并加速训练
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Use Automatic Mixed Precision (FP16) training (default: False)')

    # Learning Rate Warm-up
    # warmup_epochs: 学习率预热的 epoch 数
    # 在预热期间，学习率从 0 线性增加到设定的 lr
    # 有助于训练初期的稳定性，特别是使用大 batch size 或大学习率时
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Number of warmup epochs (default: 0, no warmup)')

    # warmup_type: 预热类型
    # - 'linear': 线性增加学习率
    # - 'cosine': 余弦曲线增加学习率
    parser.add_argument('--warmup_type', type=str, default='linear',
                        choices=['linear', 'cosine'],
                        help='Warmup schedule type (default: linear)')

    # EMA (Exponential Moving Average) 参数
    # EMA 通过维护模型参数的指数移动平均来提高推理稳定性
    # 参考: RoboticsDiffusionTransformer / DiffusionPolicy
    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='Use Exponential Moving Average for model weights (default: False)')
    parser.add_argument('--ema_inv_gamma', type=float, default=1.0,
                        help='EMA inverse gamma for warmup schedule (default: 1.0)')
    parser.add_argument('--ema_power', type=float, default=0.6667,
                        help='EMA power for warmup schedule (default: 2/3)')
    parser.add_argument('--ema_max_value', type=float, default=0.9999,
                        help='Maximum EMA decay rate (default: 0.9999)')

    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save checkpoint every N epochs (default: 50)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')

    # WandB arguments
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='robotwin_nemodit',
                        help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='WandB entity (username or team name)')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='WandB run name')

    # 图像预处理选项
    # no_resize: 跳过 Resize 和 CenterCrop，保持原始图像尺寸
    # 适用于所有相机图像尺寸一致的情况，可保留更多原始信息
    parser.add_argument('--no_resize', action='store_true', default=True,
                        help='Skip image resizing, use original size (requires same size for all cameras)')


    # 保留resize，确保使用SigLip等模型直接调用
    # 如果保留，需要更改相应的Eval文件
    # Image size
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for vision backbone (default: 224)')

    return parser.parse_args()


def prepare_dataloader(args):
    """Prepare robot dataset dataloader."""

    # Image transformations for vision backbone
    if args.no_resize:
        # 不缩放，保持原始图像尺寸，只做归一化
        # 要求所有相机图像尺寸一致
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        print("Using original image size (no resize)")
    else:
        # 标准预处理: Resize + CenterCrop + Normalize
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

    # Create dataset with n_obs_steps support
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
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    return dataloader, dataset


def create_model(args):
    """Create ActionModel with vision conditioning."""

    model = ActionModel(
        token_size=args.token_size,
        model_type=args.model_type,
        in_channels=args.action_dim,
        future_action_window_size=args.future_action_window,
        past_action_window_size=args.past_action_window,
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        use_vision_condition=True,
        vision_backbone_type=args.vision_backbone,
        vision_pretrained=args.vision_pretrained,
        num_cameras=args.num_cameras,
        freeze_vision_backbone=args.freeze_vision,
        adapter_type=args.adapter_type,
        class_dropout_prob=args.dropout_prob,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        temporal_agg=args.temporal_agg,
    )

    return model


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, warmup_type='linear'):
    """
    创建带有 warmup 的 cosine annealing 学习率调度器。

    学习率变化:
    - Warmup 阶段: 从 0 线性/余弦增加到 base_lr
    - Cosine 阶段: 从 base_lr 余弦衰减到 0

    Args:
        optimizer: 优化器
        warmup_epochs: warmup 的 epoch 数
        total_epochs: 总训练 epoch 数
        warmup_type: 'linear' 或 'cosine'

    Returns:
        scheduler: LambdaLR 调度器
    """
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # Warmup 阶段
            if warmup_type == 'linear':
                return (current_epoch + 1) / warmup_epochs
            else:  # cosine warmup
                return 0.5 * (1 - math.cos(math.pi * (current_epoch + 1) / warmup_epochs))
        else:
            # Cosine annealing 阶段
            progress = (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, args, filename=None, ema_model=None):
    """Save training checkpoint."""

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if filename is None:
        # Simple naming: {epoch}.pt to match eval script expectations
        filename = f'{epoch}.pt'

    checkpoint_path = os.path.join(args.checkpoint_dir, filename)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'args': vars(args)
    }

    # 保存 AMP scaler 状态 (如果使用)
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    # 保存 EMA 状态 (如果使用)
    if ema_model is not None:
        checkpoint['ema_state_dict'] = ema_model.state_dict()

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    # Also save as latest checkpoint
    latest_path = os.path.join(args.checkpoint_dir, 'latest.pt')
    torch.save(checkpoint, latest_path)


def load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path, ema_model=None):
    """Load training checkpoint."""

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 加载 scheduler 状态 (如果存在)
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 加载 AMP scaler 状态 (如果存在且正在使用)
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # 加载 EMA 状态 (如果存在且正在使用)
    if ema_model is not None and 'ema_state_dict' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        print(f"EMA model restored (decay={ema_model.decay:.6f}, step={ema_model.optimization_step})")

    epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']

    print(f"Resumed from epoch {epoch}, global step {global_step}")

    return epoch, global_step


def train():
    """Main training loop."""

    # Parse arguments
    args = parse_args()

    # Auto-set action_dim based on action_type and use_both_arms
    if args.action_type == 'endpose':
        # End-effector pose: 3D translation + 6D rot6d + 1D gripper = 10D per arm
        if args.use_both_arms:
            args.action_dim = 20  # (3D + 6D + 1D) * 2 = 20D for dual arm
        else:
            args.action_dim = 10  # (3D + 6D + 1D) = 10D for single arm
    else:  # joint
        # Joint angles: 6D joint + 1D gripper = 7D per arm
        if args.use_both_arms:
            args.action_dim = 14  # (6D + 1D) * 2 = 14D for dual arm
        else:
            args.action_dim = 7   # (6D + 1D) = 7D for single arm

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Action type: {args.action_type}")
    print(f"Action dimension: {args.action_dim} ({'dual arm' if args.use_both_arms else 'single arm'})")
    print(f"Temporal settings: n_obs_steps={args.n_obs_steps}, n_action_steps={args.n_action_steps}, "
          f"future_action_window={args.future_action_window}, temporal_agg={args.temporal_agg}")

    # Initialize WandB
    wandb_logger = None
    if args.use_wandb:
        print("Initializing WandB...")
        run_name = args.wandb_name if args.wandb_name else f"{os.path.basename(args.checkpoint_dir)}_{args.model_type}"
        wandb_logger = WandbLogger(
            project_name=args.wandb_project,
            run_name=run_name,
            config=vars(args),
            entity=args.wandb_entity
        )

    # Create dataloader
    print("Loading dataset...")
    dataloader, dataset = prepare_dataloader(args)
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    # Create model
    print("Creating model...")
    model = create_model(args)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler (with optional warmup)
    if args.warmup_epochs > 0:
        scheduler = get_warmup_cosine_scheduler(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            warmup_type=args.warmup_type
        )
        print(f"Using warmup: {args.warmup_epochs} epochs ({args.warmup_type})")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

    # Mixed Precision Training (AMP)
    scaler = GradScaler() if args.use_amp else None
    if args.use_amp:
        print("Using Automatic Mixed Precision (AMP) training")

    # EMA (Exponential Moving Average)
    ema_model = None
    if args.use_ema:
        ema_model = EMAModel(
            model,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            max_value=args.ema_max_value,
        )
        print(f"Using EMA (inv_gamma={args.ema_inv_gamma}, power={args.ema_power}, "
              f"max_value={args.ema_max_value})")

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        start_epoch, global_step = load_checkpoint(model, optimizer, scheduler, scaler, args.resume, ema_model)

    # Training loop
    print("Starting training...")
    model.train()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['images'].to(device)  # (B, n_obs_steps, num_cameras, 3, H, W)
            state = batch['state'].to(device)    # (B, action_dim) - 机器人当前状态
            actions = batch['actions'].to(device)  # (B, future_window - 1, action_dim)

            optimizer.zero_grad()

            if args.use_amp:
                # Mixed Precision Training
                with autocast():
                    loss = model.loss(x=actions, images=images, state=state)

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Gradient clipping (unscale first for correct clipping)
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard FP32 Training
                loss = model.loss(x=actions, images=images, state=state)

                loss.backward()

                # Gradient clipping
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

                optimizer.step()

            # Update EMA model
            if ema_model is not None:
                ema_model.step(model)

            # Update metrics
            epoch_loss += loss.item()
            global_step += 1

            # Log to WandB
            if wandb_logger:
                log_dict = {
                    'train/loss': loss.item(),
                    'train/lr': optimizer.param_groups[0]["lr"],
                    'train/epoch': epoch + 1,
                    'train/global_step': global_step,
                }
                if ema_model is not None:
                    log_dict['train/ema_decay'] = ema_model.decay
                wandb_logger.log(log_dict, step=global_step)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

        # Update learning rate
        scheduler.step()

        # Compute average epoch loss
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, global_step, args, ema_model=ema_model)

    # Save final checkpoint
    print("Training completed!")
    save_checkpoint(model, optimizer, scheduler, scaler, args.epochs, global_step, args, filename='final.pt', ema_model=ema_model)

    if wandb_logger:
        wandb_logger.finish()


if __name__ == '__main__':
    train()
