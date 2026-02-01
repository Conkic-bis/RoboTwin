import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import argparse
from pathlib import Path

from model.action_model.action_model import ActionModel
from dataloader import RobotDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DiT for robot action generation')

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to robot dataset directory containing .hdf5 files')
    parser.add_argument('--num_cameras', type=int, default=4,
                        help='Number of camera views (default: 4)')
    parser.add_argument('--use_both_arms', action='store_true', default=True,
                        help='Use both arms data (default: False)')
    parser.add_argument('--action_type', type=str, default='joint',
                        choices=['endpose', 'joint'],
                        help='Action type: endpose (ee pose) or joint (joint angles)')
    parser.add_argument('--quat_convention', type=str, default='wxyz',
                        choices=['wxyz', 'xyzw'],
                        help='Quaternion convention in HDF5 data (default: wxyz, only for endpose)')

    # Model arguments
    parser.add_argument('--model_type', type=str, default='DiT-XL',
                        choices=['DiT-S', 'DiT-B', 'DiT-L', 'DiT-XL'],
                        help='DiT model size (default: DiT-B)')
    parser.add_argument('--dropout_prob', type=float, default=0,
                        help='Class dropout probability for classifier-free guidance (default: 0.1)')
    parser.add_argument('--action_dim', type=int, default=7,
                        help='Action dimension (default: 10 for 3D translation + 6D rot6d + 1D gripper)')
    parser.add_argument('--future_action_window', type=int, default=12,
                        help='Number of future action steps to predict (default: 16)')
    parser.add_argument('--past_action_window', type=int, default=0,
                        help='Number of past action steps as context (default: 0)')
    parser.add_argument('--token_size', type=int, default=2048,
                        help='Token size for conditioning (default: 2048)')

    # n_obs_steps: 观测步数，用于视觉编码的历史帧数
    # n_obs_steps 的最后一帧对应动作序列的第一帧时刻
    parser.add_argument('--n_obs_steps', type=int, default=2,
                        help='Number of observation steps for visual conditioning (default: 2)')
    # n_action_steps: 动作执行步数，推理时实际执行的动作步数
    # 通常设置为 future_action_window 的一半或更少，用于 receding horizon control
    parser.add_argument('--n_action_steps', type=int, default=8,
                        help='Number of action steps to execute during inference (default: 8)')
    # temporal_agg: 时间聚合方式
    # - 'last': 只使用最后一帧观测
    # - 'mean': 对所有观测帧取平均
    # - 'concat': 拼接所有帧特征后投影
    parser.add_argument('--temporal_agg', type=str, default='last',
                        choices=['last', 'mean', 'concat'],
                        help='Temporal aggregation method for multi-frame observations (default: last)')

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
    parser.add_argument('--diffusion_steps', type=int, default=100,
                        help='Number of diffusion steps (default: 100)')
    parser.add_argument('--noise_schedule', type=str, default='squaredcos_cap_v2',
                        help='Noise schedule type (default: squaredcos_cap_v2)')

    # Training arguments 
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size per GPU (default: 16)')

    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs (default: 1000)')
                        
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')

    # weight_decay: 权重衰减系数，用于L2正则化防止过拟合
    # AdamW默认建议0.01，可根据数据集大小调整
    # 数据集小时可适当增大（0.01-0.1），数据集大时可减小（0.0-0.01）
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for L2 regularization (default: 0.01)')

    # grad_clip: 梯度裁剪的最大范数，防止梯度爆炸
    # 通常设置为1.0，如果训练不稳定可尝试减小到0.5
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm (default: 1.0)')

    # num_workers: 数据加载的并行工作进程数
    # 建议设置为CPU核心数的一半，最多不超过8
    # 设置过高可能导致内存问题和进程间通信开销
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')

    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--save_every', type=int, default=100,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')

    # Image size
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for vision backbone (default: 224)')

    return parser.parse_args()


def prepare_dataloader(args):
    """Prepare robot dataset dataloader."""

    # Image transformations for vision backbone
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
        n_obs_steps=args.n_obs_steps,  # 多帧观测支持
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
        temporal_agg=args.temporal_agg,
    )

    return model


def save_checkpoint(model, optimizer, epoch, global_step, args, filename=None):
    """Save training checkpoint."""

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if filename is None:
        # Simple naming: {epoch}.pt to match eval script expectations
        filename = f'{epoch}.pt'

    checkpoint_path = os.path.join(args.checkpoint_dir, filename)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'args': vars(args)
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    # Also save as latest checkpoint
    latest_path = os.path.join(args.checkpoint_dir, 'latest.pt')
    torch.save(checkpoint, latest_path)


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load training checkpoint."""

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        start_epoch, global_step = load_checkpoint(model, optimizer, args.resume)

    # Training loop
    print("Starting training...")
    model.train()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['images'].to(device)  # (B, num_cameras, 3, H, W)
            actions = batch['actions'].to(device)  # (B, future_window, action_dim)

            # Forward pass: compute loss
            loss = model.loss(x=actions, images=images)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            global_step += 1

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
            save_checkpoint(model, optimizer, epoch + 1, global_step, args)

    # Save final checkpoint
    print("Training completed!")
    save_checkpoint(model, optimizer, args.epochs, global_step, args, filename='final.pt')


if __name__ == '__main__':
    train()
