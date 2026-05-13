"""A2A training workspace for RoboTwin.

Drops every RoboVerse/Metasim/curobo-flavoured dependency from the original
``default_runner.py`` and keeps only what's needed for offline training on the
multi-camera ZARR produced by ``policy/A2A/process_data.py``. Evaluation is
performed by RoboTwin's ``script/eval_policy.py`` via the deploy_policy
interface — there is no in-workspace rollout loop.

Checkpoint schema (matches policy/DP):
    payload = {
        "cfg": OmegaConf,
        "state_dicts": {"model": ..., "ema_model": ..., "optimizer": ...},
        "pickles": {"_output_dir": ...},
    }
"""

import copy
import os
import pathlib
import random

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from a2a_flow_matching.common.json_logger import JsonLogger
from a2a_flow_matching.common.lr_scheduler import get_scheduler
from a2a_flow_matching.common.pytorch_util import optimizer_to
from a2a_flow_matching.model.diffusion.ema_model import EMAModel
from a2a_flow_matching.workspace.base_workspace import BaseWorkspace


class _BatchSampler:
    def __init__(self, data_size, batch_size, shuffle=False, seed=0, drop_last=True):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None

    def __iter__(self):
        if self.shuffle:
            perm = self.rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[: -self.discard]
        perm = perm.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch


def _create_dataloader(dataset, *, batch_size, shuffle, num_workers, pin_memory, persistent_workers, seed=0):
    batch_sampler = _BatchSampler(len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=True)

    def collate(x):
        assert len(x) == 1
        return x[0]

    return DataLoader(
        dataset,
        collate_fn=collate,
        sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
    )


class A2AWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Build the policy from cfg.policy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    # -------- training --------
    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            latest = self.get_checkpoint_path()
            if latest.is_file():
                print(f"[A2AWorkspace] resuming from {latest}")
                self.load_checkpoint(path=latest)

        dataset = hydra.utils.instantiate(cfg.task.dataset)
        train_loader = _create_dataloader(dataset, **cfg.dataloader)
        val_dataset = dataset.get_validation_dataset()
        val_loader = _create_dataloader(val_dataset, **cfg.val_dataloader)

        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        steps_per_epoch = max(len(train_loader), 1)
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(steps_per_epoch * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        ema = None
        if self.ema_model is not None:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # Optional wandb hook — best-effort; never blocks training.
        wandb_run = None
        if getattr(cfg.logging, "mode", "disabled") in ("online", "offline"):
            try:
                import wandb
                logging_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
                wandb_run = wandb.init(
                    dir=str(self.output_dir),
                    config=OmegaConf.to_container(cfg, resolve=True),
                    **logging_cfg,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[A2AWorkspace] wandb disabled: {exc}")
                wandb_run = None

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        train_sampling_batch = None
        save_root = getattr(cfg.checkpoint, "save_root_dir", None)

        with JsonLogger(log_path) as json_logger:
            for _ in range(cfg.training.num_epochs):
                step_log = {}

                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = []
                with tqdm.tqdm(
                    train_loader,
                    desc=f"epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dataset.postprocess(batch, device)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        if ema is not None:
                            ema.step(self.model)

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last = batch_idx == (len(train_loader) - 1)
                        if not is_last:
                            if wandb_run is not None:
                                wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (
                            cfg.training.max_train_steps is not None
                            and batch_idx >= cfg.training.max_train_steps - 1
                        ):
                            break

                step_log["train_loss"] = float(np.mean(train_losses)) if train_losses else 0.0

                policy = self.ema_model if self.ema_model is not None else self.model
                policy.eval()

                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = []
                        for batch_idx, batch in enumerate(val_loader):
                            batch = dataset.postprocess(batch, device)
                            val_losses.append(self.model.compute_loss(batch).item())
                            if (
                                cfg.training.max_val_steps is not None
                                and batch_idx >= cfg.training.max_val_steps - 1
                            ):
                                break
                        if val_losses:
                            step_log["val_loss"] = float(np.mean(val_losses))

                # Periodic checkpoint
                ckpt_every = cfg.training.checkpoint_every
                last_epoch = self.epoch + 1 >= cfg.training.num_epochs
                if ((self.epoch + 1) % ckpt_every == 0) or last_epoch:
                    if save_root is None:
                        ckpt_path = pathlib.Path(self.output_dir).joinpath(
                            "checkpoints", f"{self.epoch + 1}.ckpt"
                        )
                    else:
                        ckpt_path = pathlib.Path(save_root).joinpath(
                            "checkpoints", f"{self.epoch + 1}.ckpt"
                        )
                    self.save_checkpoint(str(ckpt_path))

                policy.train()
                json_logger.log(step_log)
                if wandb_run is not None:
                    wandb_run.log(step_log, step=self.global_step)
                self.global_step += 1
                self.epoch += 1

        if wandb_run is not None:
            wandb_run.finish()
