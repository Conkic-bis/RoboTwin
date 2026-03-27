import torch
import argparse
import numpy as np
from pathlib import Path
from torchvision import transforms

from model.action_model.action_model import ActionModel
from dataloader import RobotDataset


def parse_args():
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate DiT Action Model')

    # Checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')

    # Data arguments (for loading test samples)
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to robot dataset for evaluation (optional)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to evaluate (default: 5)')

    # Inference arguments
    parser.add_argument('--ddim_steps', type=int, default=10,
                        help='DDIM sampling steps (default: 10)')
    parser.add_argument('--cfg_scale', type=float, default=1.0,
                        help='Classifier-free guidance scale (default: 1.0, no guidance)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')

    return parser.parse_args()


def load_model(checkpoint_path, device):
    """
    从 checkpoint 加载模型。

    Args:
        checkpoint_path: checkpoint 文件路径
        device: 计算设备

    Returns:
        model: 加载好权重的 ActionModel
        args: 训练时的参数
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 获取训练时的参数
    train_args = checkpoint['args']
    print(f"Model config: {train_args['model_type']}, action_dim={train_args['action_dim']}")
    print(f"Temporal config: n_obs_steps={train_args['n_obs_steps']}, "
          f"n_action_steps={train_args['n_action_steps']}, "
          f"future_action_window={train_args['future_action_window']}")

    # 创建模型
    model = ActionModel(
        token_size=train_args['token_size'],
        model_type=train_args['model_type'],
        in_channels=train_args['action_dim'],
        future_action_window_size=train_args['future_action_window'],
        past_action_window_size=train_args['past_action_window'],
        diffusion_steps=train_args['diffusion_steps'],
        noise_schedule=train_args['noise_schedule'],
        use_vision_condition=True,
        vision_backbone_type=train_args['vision_backbone'],
        vision_pretrained=False, 
        num_cameras=train_args['num_cameras'],
        freeze_vision_backbone=False,
        adapter_type=train_args['adapter_type'],
        class_dropout_prob=0.0, 
        n_obs_steps=train_args['n_obs_steps'],
        n_action_steps=train_args['n_action_steps'],
        temporal_agg=train_args['temporal_agg'],
    )

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from epoch {checkpoint['epoch']}")
    return model, train_args


def prepare_eval_dataloader(data_path, train_args):
    """
    准备评估数据集。

    Args:
        data_path: 数据集路径
        train_args: 训练参数

    Returns:
        dataset: RobotDataset
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(train_args['image_size']),
        transforms.CenterCrop(train_args['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    dataset = RobotDataset(
        data_path=data_path,
        future_action_window=train_args['future_action_window'],
        past_action_window=train_args['past_action_window'],
        transform=transform,
        num_cameras=train_args['num_cameras'],
        use_both_arms=train_args.get('use_both_arms', False),
        action_type=train_args.get('action_type', 'endpose'),
        quat_convention=train_args.get('quat_convention', 'wxyz'),
        n_obs_steps=train_args['n_obs_steps'],
    )

    return dataset


@torch.no_grad()
def evaluate_sample(model, images, state, gt_actions, ddim_steps, cfg_scale, device):
    """
    对单个样本进行评估。

    Args:
        model: ActionModel
        images: (1, n_obs_steps, num_cameras, C, H, W) 观测图像
        state: (1, action_dim) 机器人当前状态
        gt_actions: (1, future_action_window - 1, action_dim) ground truth 动作 (不含 state)
        ddim_steps: DDIM 采样步数
        cfg_scale: CFG scale
        device: 计算设备

    Returns:
        pred_actions: (n_action_steps, action_dim) 预测的动作
        metrics: 评估指标字典
    """
    images = images.to(device)
    state = state.to(device)
    gt_actions = gt_actions.to(device)

    # 推理生成动作 (with state conditioning)
    pred_actions = model.sample(
        images,
        state=state,
        ddim_steps=ddim_steps,
        cfg_scale=cfg_scale,
        return_all=False
    )  # (1, n_action_steps, action_dim)

    # 计算 MSE (只比较 n_action_steps 步)
    n_action_steps = pred_actions.shape[1]
    gt_truncated = gt_actions[:, :n_action_steps, :]
    mse = ((pred_actions - gt_truncated) ** 2).mean().item()

    # 计算 L1 误差
    l1 = (pred_actions - gt_truncated).abs().mean().item()

    metrics = {
        'mse': mse,
        'l1': l1,
    }

    return pred_actions.squeeze(0).cpu().numpy(), metrics


def main():
    args = parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    model, train_args = load_model(args.checkpoint, device)

    # 如果没有指定数据路径，只展示推理接口
    if args.data_path is None:
        print("\n" + "="*60)
        print("推理接口示例 (Inference API Example)")
        print("="*60)
        print("""
# 1. 加载模型
model, train_args = load_model('checkpoints/100.pt', device)

# 2. 准备观测图像和当前状态
# images shape: (batch_size, n_obs_steps, num_cameras, 3, H, W)
# 例如: (1, 2, 4, 3, 224, 224) 表示 batch=1, 2帧观测, 4个相机
# state shape: (batch_size, action_dim)
# 例如: (1, 14) 表示机器人当前关节状态

# 3. 推理生成动作 (state 作为条件输入，不参与噪音扩散)
actions = model.sample(
    images,
    state=state,        # 机器人当前状态
    ddim_steps=10,      # DDIM 采样步数
    cfg_scale=1.0,      # CFG scale (1.0 = 无 guidance)
    return_all=False    # 只返回 n_action_steps 步
)
# actions shape: (batch_size, n_action_steps, action_dim)
# 注意: 预测的是 state 之后的动作，不含 state 本身

# 4. 执行动作 (receding horizon control)
for i in range(n_action_steps):
    action = actions[0, i]  # 取第 i 步动作
    robot.execute(action)   # 执行
""")
        return

    # 加载评估数据
    print(f"\nLoading evaluation data from: {args.data_path}")
    dataset = prepare_eval_dataloader(args.data_path, train_args)
    print(f"Dataset size: {len(dataset)}")

    # 评估指定数量的样本
    num_samples = min(args.num_samples, len(dataset))
    all_mse = []
    all_l1 = []

    print(f"\nEvaluating {num_samples} samples...")
    print("-" * 60)

    for i in range(num_samples):
        sample = dataset[i]
        images = sample['images'].unsqueeze(0)  # (1, n_obs_steps, num_cameras, C, H, W)
        state = sample['state'].unsqueeze(0)    # (1, action_dim)
        gt_actions = sample['actions'].unsqueeze(0)  # (1, future_window - 1, action_dim)

        pred_actions, metrics = evaluate_sample(
            model, images, state, gt_actions,
            args.ddim_steps, args.cfg_scale, device
        )

        all_mse.append(metrics['mse'])
        all_l1.append(metrics['l1'])

        print(f"Sample {i+1}: MSE={metrics['mse']:.6f}, L1={metrics['l1']:.6f}")
        print(f"  Pred shape: {pred_actions.shape}")
        print(f"  Pred[0]: {pred_actions[0]}")

    # 汇总统计
    print("-" * 60)
    print(f"Average MSE: {np.mean(all_mse):.6f} (+/- {np.std(all_mse):.6f})")
    print(f"Average L1:  {np.mean(all_l1):.6f} (+/- {np.std(all_l1):.6f})")


if __name__ == '__main__':
    main()
