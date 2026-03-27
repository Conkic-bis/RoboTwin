# Reference: RoboticsDiffusionTransformer [https://github.com/thu-ml/RoboticsDiffusionTransformer]
# Reference: DiffusionPolicy [https://github.com/real-stanford/diffusion_policy]

import copy
import torch
from torch.nn.modules.batchnorm import _BatchNorm


class EMAModel:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy of the model whose parameters are updated as an
    exponential moving average of the online (trained) model.  The EMA model
    typically produces smoother and more stable predictions, which is
    especially beneficial for diffusion-based policies.

    Usage:
        ema = EMAModel(model)          # create shadow copy
        ...
        optimizer.step()               # normal training step
        ema.step(model)                # update EMA weights
        ...
        ema.averaged_model             # use for inference / checkpointing
    """

    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999,
    ):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average.
            gamma=1, power=2/3 are good values for models you plan to train
            for a million or more steps (reaches decay factor 0.999 at 31.6K
            steps, 0.9999 at 1M steps).
            gamma=1, power=3/4 for models you plan to train for less (reaches
            decay factor 0.999 at 10K steps, 0.9999 at 215.4k steps).

        Args:
            model: The model whose parameters will be averaged.
            update_after_step (int): Start EMA updates only after this many
                optimisation steps.  Before that the decay is forced to 0
                (i.e. the EMA model mirrors the online model exactly).
            inv_gamma (float): Inverse multiplicative factor of EMA warmup.
            power (float): Exponential factor of EMA warmup.
            min_value (float): Minimum EMA decay rate.
            max_value (float): Maximum EMA decay rate.
        """
        self.averaged_model = copy.deepcopy(model)
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """Compute the decay factor for the exponential moving average."""
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        """Update EMA parameters from *new_model*."""
        self.decay = self.get_decay(self.optimization_step)

        for module, ema_module in zip(
            new_model.modules(), self.averaged_model.modules()
        ):
            for param, ema_param in zip(
                module.parameters(recurse=False),
                ema_module.parameters(recurse=False),
            ):
                if isinstance(param, dict):
                    raise RuntimeError("Dict parameter not supported")

                if isinstance(module, _BatchNorm) or not param.requires_grad:
                    # Copy BatchNorm stats and frozen params directly
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(
                        param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay
                    )

        self.optimization_step += 1

    def state_dict(self):
        """Return serialisable state for checkpointing."""
        return {
            "averaged_model_state_dict": self.averaged_model.state_dict(),
            "decay": self.decay,
            "optimization_step": self.optimization_step,
        }

    def load_state_dict(self, state_dict):
        """Restore from a previously saved *state_dict*."""
        self.averaged_model.load_state_dict(state_dict["averaged_model_state_dict"])
        self.decay = state_dict["decay"]
        self.optimization_step = state_dict["optimization_step"]
