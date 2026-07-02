"""Train Mu0 trace-conditioned action policy on RoboCasa demonstrations.

The public config surface is intentionally small. Removed action-decoder
options are fixed to the values from the original RoboCasa trace run:
Gemma scratch action expert, language prefix enabled, no depth predictor,
uniform 20x20 keypoints, spatial gripper tokens, and partial trace guidance.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed as accelerate_set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.multi_dataset import MultiLeRobotDataset
from lerobot.datasets.trace_delta_stats import load_trace_stats
from lerobot.policies.mu0.mu0_policy import GRID_SIZE, Mu0Policy, Mu0PolicyConfig
from lerobot.utils.utils import cycle


@dataclass
class TrainMu0PolicyConfig:
    trace_checkpoint: str = ""
    delta_stats_path: str = ""
    dataset_repo_ids: list[str] = field(default_factory=list)
    dataset_roots: list[str] = field(default_factory=list)
    task_names: list[str] = field(default_factory=list)
    output_dir: str = "outputs/robocasa/mu0_policy"
    debug_overlay_dir: str | None = None
    debug_overlay_every_n: int = 1

    camera_name: str = "observation.images.robot0_agentview_left"
    gripper_camera_name: str = "observation.images.robot0_eye_in_hand"
    num_episodes: int = -1

    action_dim: int = 12
    chunk_size: int = 16
    state_dim: int = 16
    gripper_spatial_pool_size: int = 8
    use_trace: bool = True

    training_steps: int = 60000
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    warmup_steps: int = 1000
    log_freq: int = 10
    save_freq: int = 5000
    val_l1_freq: int = 1500
    num_workers: int = 8
    resume_from: str = ""

    wandb_enable: bool = True
    wandb_project: str = "robocasa"
    wandb_entity: str | None = "tracegenv2"
    seed: int = 42
    job_name: str | None = None

    def build_policy_config(
        self,
        delta_scale: tuple[float, float, float] | None = None,
    ) -> Mu0PolicyConfig:
        return Mu0PolicyConfig(
            action_dim=self.action_dim,
            chunk_size=self.chunk_size,
            state_dim=self.state_dim,
            gripper_spatial_pool_size=self.gripper_spatial_pool_size,
            use_trace=self.use_trace,
            debug_overlay_dir=self.debug_overlay_dir,
            debug_overlay_every_n=self.debug_overlay_every_n,
            delta_scale=delta_scale,
        )


def _log_trainable_summary(model: torch.nn.Module, max_depth: int = 2) -> None:
    def count(module: torch.nn.Module) -> tuple[int, int]:
        total = sum(param.numel() for param in module.parameters())
        trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
        return total, trainable

    def fmt(num_params: int) -> str:
        return f"{num_params / 1e6:9.2f}M" if num_params >= 1_000_000 else f"{num_params:11,}"

    def walk(module: torch.nn.Module, name: str, depth: int) -> None:
        total, trainable = count(module)
        if total == 0:
            return
        logging.info(
            "%s%-55s trainable=%s / total=%s  (%5.1f%%)",
            "  " * depth,
            name,
            fmt(trainable),
            fmt(total),
            100 * trainable / max(total, 1),
        )
        if depth < max_depth:
            for child_name, child in module.named_children():
                walk(child, f"{name}.{child_name}", depth + 1)

    logging.info("=" * 100)
    logging.info("Trainable parameter summary")
    logging.info("=" * 100)
    walk(model, "model", 0)
    total, trainable = count(model)
    logging.info("=" * 100)
    logging.info(
        "TOTAL trainable=%s / total=%s  (%.2f%%)",
        fmt(trainable),
        fmt(total),
        100 * trainable / max(total, 1),
    )
    logging.info("=" * 100)


def _prepare_batch_for_trace_model(
    batch: dict[str, torch.Tensor],
    tokenizer,
    tokenizer_max_length: int,
    history_len: int,
    device: torch.device,
    camera_name: str,
    gripper_camera_name: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a RoboCasa LeRobot batch to the Mu0Policy inputs."""
    from lerobot.datasets.trace_dataset import (
        CURRENT_KP_KEY,
        IMAGE_KEY,
        KEY_PAD_MASK_KEY,
        TRAJ_HISTORY_KEY,
    )
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    batch_size = batch["action"].shape[0]
    trace_batch: dict[str, torch.Tensor] = {}
    if camera_name == gripper_camera_name:
        raise ValueError(
            "camera_name and gripper_camera_name must point to different cameras; "
            f"both were {camera_name!r}."
        )

    def latest_image(image_key: str, role: str) -> torch.Tensor:
        if image_key not in batch:
            available = [key for key in batch if key.startswith("observation.images.")]
            raise ValueError(f"{role} camera {image_key!r} not in batch. Available: {available}")
        image = batch[image_key]
        if image.ndim == 5:
            image = image[:, -1]
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                f"{role} camera {image_key!r} must have shape (B, 3, H, W) "
                f"or (B, T, 3, H, W); got {tuple(image.shape)}."
            )
        return image.to(device)

    trace_batch[IMAGE_KEY] = latest_image(camera_name, "Third-view")

    tasks = batch.get("task", None)
    if isinstance(tasks, (list, tuple)):
        task_strs = [
            f"Task: {task.strip()}\n" if isinstance(task, str) else "Task: manipulate object\n"
            for task in tasks
        ]
    else:
        task_strs = ["Task: manipulate object\n"] * batch_size
    tok_out = tokenizer(
        task_strs,
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    trace_batch[OBS_LANGUAGE_TOKENS] = tok_out["input_ids"].to(device)
    trace_batch[OBS_LANGUAGE_ATTENTION_MASK] = tok_out["attention_mask"].bool().to(device)

    u = torch.linspace(-1.0, 1.0, GRID_SIZE, device=device)
    v = torch.linspace(-1.0, 1.0, GRID_SIZE, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")
    grid_uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)
    num_kp = grid_uv.shape[0]
    current_kp = torch.zeros(batch_size, num_kp, 3, device=device)
    current_kp[:, :, :2] = grid_uv
    trace_batch[CURRENT_KP_KEY] = current_kp
    trace_batch[TRAJ_HISTORY_KEY] = torch.zeros(batch_size, num_kp, history_len, 3, device=device)
    trace_batch[KEY_PAD_MASK_KEY] = torch.ones(batch_size, num_kp, dtype=torch.bool, device=device)

    state_key = "observation.state"
    if state_key in batch:
        proprio = batch[state_key]
        if proprio.ndim == 3:
            proprio = proprio[:, -1]
        proprioception = proprio.float().to(device)
    else:
        proprioception = torch.zeros(batch_size, 16, device=device)

    gt_actions = batch["action"].float().to(device)
    if gt_actions.ndim == 2:
        gt_actions = gt_actions.unsqueeze(1)
    gripper_image = latest_image(gripper_camera_name, "Gripper")
    return trace_batch, proprioception, gt_actions, gripper_image


def _trainable_state_dict(model: Mu0Policy) -> dict[str, torch.Tensor]:
    trainable_prefixes = (
        "action_decoder.",
        "gated_mixer.",
        "gripper_proj.",
        "gripper_spatial_pos_embed",
    )
    return {
        key: value
        for key, value in model.state_dict().items()
        if any(key.startswith(prefix) for prefix in trainable_prefixes)
    }


def _required_checkpoint_prefixes(use_trace: bool) -> tuple[str, ...]:
    prefixes = [
        "gripper_proj.",
        "gripper_spatial_pos_embed",
        "action_decoder.gemma_expert.model.layers.0.",
        "action_decoder.gemma_expert.model.norm.",
        "action_decoder.action_in_proj.",
        "action_decoder.action_out_proj.",
        "action_decoder.time_mlp_in.",
        "action_decoder.time_mlp_out.",
        "action_decoder.z_norm.",
        "action_decoder.z_proj.",
        "action_decoder.lang_norm.",
        "action_decoder.lang_proj.",
        "action_decoder.visual_norm.",
        "action_decoder.visual_proj.",
        "action_decoder.state_proj.",
    ]
    if use_trace:
        prefixes.append("gated_mixer.")
    return tuple(prefixes)


def _save_checkpoint(
    *,
    step: int,
    output_dir: Path,
    model: Mu0Policy,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    ckpt_dir = output_dir / f"step_{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": _trainable_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        ckpt_dir / "checkpoint.pt",
    )
    logging.info("Saved checkpoint: %s", ckpt_dir)


@draccus.wrap()
def train(cfg: TrainMu0PolicyConfig) -> None:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision="bf16")
    is_main = accelerator.is_main_process
    device = accelerator.device
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.set_device(device)

    if is_main:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print(
        f"[rank {accelerator.process_index}/{accelerator.num_processes}] "
        f"device={accelerator.device} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}",
        flush=True,
    )
    accelerate_set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.json").open("w") as f:
            json.dump(draccus.encode(cfg), f, indent=2)
        logging.info("Output dir: %s (world_size=%s)", output_dir, accelerator.num_processes)

    delta_scale = None
    if cfg.delta_stats_path:
        trace_stats = load_trace_stats(cfg.delta_stats_path)
        delta_scale = trace_stats.delta_scale
        if is_main:
            logging.info("Loaded delta_stats: delta_scale=%s", delta_scale)

    if is_main:
        logging.info("Loading trace checkpoint: %s", cfg.trace_checkpoint)
    model = Mu0Policy(
        trace_checkpoint_path=cfg.trace_checkpoint,
        config=cfg.build_policy_config(delta_scale=delta_scale),
        trace_device=device,
    )

    if is_main:
        _log_trainable_summary(model)

    from transformers import AutoTokenizer

    pcfg = model.trace_policy.config
    tokenizer = AutoTokenizer.from_pretrained(pcfg.vlm_model_name)
    tokenizer.padding_side = "right"
    tokenizer_max_length = pcfg.tokenizer_max_length
    history_len = pcfg.history_len

    if not cfg.dataset_repo_ids:
        raise ValueError("Provide at least one entry in dataset_repo_ids.")
    if len(cfg.dataset_roots) != len(cfg.dataset_repo_ids):
        raise ValueError(
            f"dataset_roots length ({len(cfg.dataset_roots)}) must match "
            f"dataset_repo_ids length ({len(cfg.dataset_repo_ids)})."
        )

    first_meta = LeRobotDatasetMetadata(cfg.dataset_repo_ids[0], root=cfg.dataset_roots[0])
    fps = first_meta.fps
    delta_timestamps = {
        "observation.state": [0.0],
        "action": [idx / fps for idx in range(cfg.chunk_size)],
        cfg.camera_name: [0.0],
        cfg.gripper_camera_name: [0.0],
    }

    episodes_dict = None
    if cfg.num_episodes > 0:
        episodes_dict = {repo_id: list(range(cfg.num_episodes)) for repo_id in cfg.dataset_repo_ids}
        if is_main:
            logging.info(
                "Using first %s episodes per task (%s tasks)",
                cfg.num_episodes,
                len(cfg.dataset_repo_ids),
            )

    dataset = MultiLeRobotDataset(
        repo_ids=cfg.dataset_repo_ids,
        roots=cfg.dataset_roots,
        delta_timestamps=delta_timestamps,
        episodes=episodes_dict,
        video_backend="pyav",
    )
    if is_main:
        logging.info(
            "Dataset: %s episodes, %s frames across %s tasks",
            dataset.num_episodes,
            dataset.num_frames,
            len(cfg.dataset_repo_ids),
        )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(cfg.warmup_steps, 1)
        progress = (step - cfg.warmup_steps) / max(cfg.training_steps - cfg.warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    dl_iter = cycle(dataloader)
    unwrapped: Mu0Policy = accelerator.unwrap_model(model)
    unwrapped.trace_model.eval()

    start_step = 1
    resume_wandb_id: str | None = None
    if cfg.resume_from:
        resume_path = Path(cfg.resume_from)
        if resume_path.is_dir():
            resume_path = resume_path / "checkpoint.pt"
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume_from checkpoint not found: {resume_path}")
        if is_main:
            logging.info("Resuming from %s", resume_path)

        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        missing, unexpected = unwrapped.load_state_dict(ckpt["model_state_dict"], strict=False)
        if is_main:
            logging.info(
                "Loaded checkpoint: %s trainable tensors, %s expected-missing, %s unexpected",
                len(ckpt["model_state_dict"]),
                len(missing),
                len(unexpected),
            )
            if unexpected:
                logging.warning("Unexpected checkpoint keys: %s", unexpected[:8])

        loaded_keys = set(ckpt["model_state_dict"].keys())
        for prefix in _required_checkpoint_prefixes(cfg.use_trace):
            if not any(key.startswith(prefix) for key in loaded_keys):
                raise RuntimeError(
                    f"Resume checkpoint missing keys with prefix {prefix!r}. "
                    "Start a fresh run or resume from a checkpoint from this simplified policy."
                )

        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            for _ in range(ckpt["step"]):
                scheduler.step()
        start_step = ckpt["step"] + 1

        wandb_id_file = output_dir / "wandb_run_id.txt"
        if wandb_id_file.is_file():
            resume_wandb_id = wandb_id_file.read_text().strip() or None

    wandb_run = None
    if cfg.wandb_enable and is_main:
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.job_name
            or f"mu0_policy_{'_'.join(cfg.task_names) if cfg.task_names else 'multitask'}",
            config=draccus.encode(cfg),
            id=resume_wandb_id,
            resume="allow" if resume_wandb_id else None,
        )
        (output_dir / "wandb_run_id.txt").write_text(wandb_run.id)

    if is_main:
        logging.info("Starting training for %s steps", cfg.training_steps)
    model.train()
    unwrapped.trace_model.eval()

    loss_sum = 0.0
    loss_count = 0
    progbar = tqdm(
        range(start_step, cfg.training_steps + 1),
        desc="train",
        initial=start_step - 1,
        total=cfg.training_steps,
        disable=not is_main,
    )

    for step in progbar:
        batch = next(dl_iter)
        trace_batch, proprioception, gt_actions, gripper_image = _prepare_batch_for_trace_model(
            batch,
            tokenizer,
            tokenizer_max_length,
            history_len,
            device,
            camera_name=cfg.camera_name,
            gripper_camera_name=cfg.gripper_camera_name,
        )
        if gt_actions.shape[1] < cfg.chunk_size:
            raise ValueError(
                f"Dataset returned only {gt_actions.shape[1]} action steps, "
                f"but chunk_size={cfg.chunk_size}."
            )

        output = model(
            trace_batch,
            proprioception,
            gripper_image=gripper_image,
            actions=gt_actions[:, : cfg.chunk_size],
        )
        loss = output["loss"]

        optimizer.zero_grad()
        accelerator.backward(loss)
        if cfg.grad_clip_norm > 0:
            accelerator.clip_grad_norm_(trainable_params, cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        loss_sum += loss.detach().item()
        loss_count += 1
        del output, loss

        if step % cfg.log_freq == 0 and is_main:
            avg_loss = loss_sum / max(loss_count, 1)
            lr = scheduler.get_last_lr()[0]
            progbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
            if wandb_run:
                wandb_run.log({"train/loss": avg_loss, "train/lr": lr}, step=step)
            loss_sum = 0.0
            loss_count = 0

        if step % cfg.val_l1_freq == 0 and is_main:
            unwrapped.eval()
            unwrapped.trace_model.eval()
            with torch.no_grad():
                val_output = unwrapped(
                    trace_batch,
                    proprioception,
                    gripper_image=gripper_image,
                    actions=None,
                )
            pred_actions = val_output["actions"]
            gt_for_val = gt_actions[:, : cfg.chunk_size]
            l1_per_dim = (pred_actions - gt_for_val).abs().mean(dim=(0, 1))
            l1_scalar = l1_per_dim.mean().item()
            logging.info(
                "[val L1] step=%s mean=%.4f per_dim=%s",
                step,
                l1_scalar,
                [f"{value:.3f}" for value in l1_per_dim.tolist()],
            )
            if wandb_run:
                wandb_run.log({"val/L1_actions": l1_scalar}, step=step)
            unwrapped.train()
            unwrapped.trace_model.eval()

        if step % cfg.save_freq == 0:
            accelerator.wait_for_everyone()
            if is_main:
                _save_checkpoint(
                    step=step,
                    output_dir=output_dir,
                    model=unwrapped,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

    accelerator.wait_for_everyone()
    if is_main:
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": cfg.training_steps,
                "model_state_dict": _trainable_state_dict(unwrapped),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            final_dir / "checkpoint.pt",
        )
        logging.info("Training complete. Final checkpoint: %s", final_dir)

    if wandb_run:
        wandb_run.finish()


def main() -> None:
    train()


if __name__ == "__main__":
    main()
