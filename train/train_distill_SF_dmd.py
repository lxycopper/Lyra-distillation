#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-Forcing-style DMD distillation trainer for Lyra-2.

This script follows the training pattern used in Self-Forcing-Plus DMD:
1) Alternating optimization between generator and critic.
2) Generator loss uses distribution-matching gradient regression:
      0.5 * ||x - (x - grad)||^2
   where grad ~= fake_score - real_score, normalized by |x - real_score|.
3) Critic loss uses denoising/flow supervision on noisy generated samples.

Adaptation notes for Lyra-2:
- real_score := frozen teacher Lyra2 model.
- fake_score := trainable Lyra2 critic model.
- generator := trainable Lyra2 student model (DMD inference path).
- "x" is generated latent tail chunk in Lyra latent space.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from lyra_2._ext.imaginaire.lazy_config import instantiate
from lyra_2._ext.imaginaire.utils import log, misc
from lyra_2._ext.imaginaire.utils.config_helper import get_config_module, override
from lyra_2._src.models.lyra2_model import WAN2PT1_I2V_COND_LATENT_KEY
from lyra_2._src.modules.conditioner import DataType
from lyra_2._src.utils.model_loader import load_model_from_checkpoint


def _clone_batch(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.clone()
    if isinstance(x, dict):
        return {k: _clone_batch(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clone_batch(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_clone_batch(v) for v in x)
    return copy.deepcopy(x)


def _ensure_neg_t5(data_batch: dict[str, Any]) -> torch.Tensor:
    neg = data_batch.get("neg_t5_text_embeddings", None)
    pos = data_batch.get("t5_text_embeddings", None)
    if neg is not None:
        return neg
    if pos is None:
        raise KeyError("Missing 't5_text_embeddings' in batch.")
    return torch.zeros_like(pos)


def _set_trainable_params(model: torch.nn.Module, train_lora_only: bool) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if train_lora_only:
            keep = ("lora" in name.lower()) or ("adapter" in name.lower())
            p.requires_grad = bool(keep)
        if p.requires_grad:
            params.append(p)
    return params


def _build_dataloader(config_file: str, experiment: str, experiment_opts: list[str]):
    config_module = get_config_module(config_file)
    cfg = importlib.import_module(config_module).make_config()
    cfg = override(cfg, ["--", f"experiment={experiment}"] + experiment_opts)
    cfg.validate()
    cfg.freeze()  # type: ignore
    return instantiate(cfg.dataloader_train), cfg


def _build_condition_pair(model, data_batch: dict[str, Any]):
    condition, uncondition = model.conditioner.get_condition_with_negative_prompt(data_batch)
    condition = condition.edit_data_type(DataType.VIDEO)
    uncondition = uncondition.edit_data_type(DataType.VIDEO)
    return condition, uncondition


def _sample_noisy_tail(model, clean_tail: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample noisy tail and return (xt_tail, vt_tail_target, timesteps_B1)."""
    b = int(clean_tail.shape[0])
    t_b = model.rectified_flow.sample_train_time(b).to(**model.flow_matching_kwargs)
    t_b = t_b.reshape(b, 1)
    timesteps = model.rectified_flow.get_discrete_timestamp(t_b, model.flow_matching_kwargs)
    sigmas = model.rectified_flow.get_sigmas(timesteps, model.flow_matching_kwargs).reshape(b, 1)
    eps = torch.randn_like(clean_tail, dtype=model.flow_matching_kwargs["dtype"])
    xt_tail, vt_tail = model.rectified_flow.get_interpolation(
        eps.to(dtype=torch.float32), clean_tail.to(dtype=torch.float32), sigmas
    )
    return xt_tail.to(dtype=clean_tail.dtype), vt_tail.to(dtype=clean_tail.dtype), timesteps.reshape(b, 1)


def _prepare_model_inputs(model, data_batch: dict[str, Any], seed_step: int):
    """Tokenize batch and build common tensors for generator/critic/teacher."""
    _, x0_latents, _ = model.get_data_and_condition(data_batch, dropout=False)
    t_hist = int(model.framepack_total_max_num_latent_frames - model.framepack_num_new_latent_frames)
    hist = x0_latents[:, :, :t_hist]
    cond_latent = data_batch[WAN2PT1_I2V_COND_LATENT_KEY]
    cond, uncond = _build_condition_pair(model, data_batch)

    tail = model.inference_dmd(
        history_latents=hist,
        cond_latent=cond_latent,
        guidance=1.0,
        seed=seed_step,
        num_steps=4,
        shift=5.0,
        t5_text_embeddings=data_batch["t5_text_embeddings"],
        neg_t5_text_embeddings=_ensure_neg_t5(data_batch),
        last_hist_frame=data_batch["last_hist_frame"],
        cond_latent_mask=data_batch["cond_latent_mask"],
        cond_latent_buffer=data_batch.get("cond_latent_buffer", None),
        fps=data_batch.get("fps", None),
        padding_mask=data_batch.get("padding_mask", None),
    )
    return x0_latents, hist, t_hist, cond, uncond, tail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Forcing style DMD trainer for Lyra-2")
    parser.add_argument("--train_config", type=str, default="", help="YAML config path for this trainer.")
    parser.add_argument("--experiment", type=str, default="lyra2")
    parser.add_argument("--config_file", type=str, default="lyra_2/_src/configs/config.py")
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--student_ckpt", type=str, default="")
    parser.add_argument("--critic_ckpt", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs/distill_sf_dmd")
    parser.add_argument("--experiment_opt", action="append", default=[], help="Hydra override, repeatable")

    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--lr_g", type=float, default=1e-5)
    parser.add_argument("--lr_d", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_g", type=float, default=1.0)
    parser.add_argument("--grad_clip_d", type=float, default=1.0)
    parser.add_argument("--train_lora_only", action="store_true")

    parser.add_argument("--generator_update_every", type=int, default=2)
    parser.add_argument("--dmd_weight", type=float, default=1.0)
    parser.add_argument("--critic_weight", type=float, default=1.0)
    parser.add_argument("--normalizer_eps", type=float, default=1e-6)

    parser.add_argument("--teacher_self_aug", action="store_true")
    parser.add_argument("--student_self_aug", action="store_true", default=True)
    parser.add_argument("--critic_self_aug", action="store_true")
    args = parser.parse_args()

    # Optional YAML override for trainer-specific arguments.
    if args.train_config:
        cfg_obj = OmegaConf.to_container(OmegaConf.load(args.train_config), resolve=True)
        if not isinstance(cfg_obj, dict):
            raise ValueError(f"train_config must be a dict-like YAML, got: {type(cfg_obj)}")
        for k, v in cfg_obj.items():
            if hasattr(args, k):
                setattr(args, k, v)
            else:
                raise KeyError(f"Unknown trainer config key: {k}")
    return args


def main() -> None:
    args = parse_args()
    misc.set_random_seed(seed=args.seed, by_rank=False)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "train_log.jsonl"

    train_loader, _ = _build_dataloader(args.config_file, args.experiment, args.experiment_opt)
    data_iter = iter(train_loader)

    # real_score (frozen teacher)
    teacher, _ = load_model_from_checkpoint(
        experiment_name=args.experiment,
        checkpoint_path=args.teacher_ckpt,
        config_file=args.config_file,
        enable_fsdp=False,
        instantiate_ema=False,
        seed=args.seed,
        experiment_opts=args.experiment_opt,
        strict=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # generator (student)
    student_ckpt = args.student_ckpt if args.student_ckpt else args.teacher_ckpt
    student, _ = load_model_from_checkpoint(
        experiment_name=args.experiment,
        checkpoint_path=student_ckpt,
        config_file=args.config_file,
        enable_fsdp=False,
        instantiate_ema=False,
        seed=args.seed,
        experiment_opts=args.experiment_opt,
        strict=True,
    )
    student.train()

    # fake_score (critic)
    critic_ckpt = args.critic_ckpt if args.critic_ckpt else student_ckpt
    critic, _ = load_model_from_checkpoint(
        experiment_name=args.experiment,
        checkpoint_path=critic_ckpt,
        config_file=args.config_file,
        enable_fsdp=False,
        instantiate_ema=False,
        seed=args.seed,
        experiment_opts=args.experiment_opt,
        strict=True,
    )
    critic.train()

    teacher.config.self_aug_enabled = bool(args.teacher_self_aug)
    student.config.self_aug_enabled = bool(args.student_self_aug)
    critic.config.self_aug_enabled = bool(args.critic_self_aug)

    trainable_g = _set_trainable_params(student, train_lora_only=bool(args.train_lora_only))
    trainable_d = _set_trainable_params(critic, train_lora_only=bool(args.train_lora_only))
    if len(trainable_g) == 0:
        raise RuntimeError("No trainable generator parameters selected.")
    if len(trainable_d) == 0:
        raise RuntimeError("No trainable critic parameters selected.")

    optimizer_g = torch.optim.AdamW(trainable_g, lr=args.lr_g, weight_decay=args.weight_decay)
    optimizer_d = torch.optim.AdamW(trainable_d, lr=args.lr_d, weight_decay=args.weight_decay)

    log.info(
        "Starting Self-Forcing style DMD distillation: "
        f"steps={args.max_steps}, "
        f"trainable_g={sum(p.numel() for p in trainable_g):,}, "
        f"trainable_d={sum(p.numel() for p in trainable_d):,}"
    )

    run_g = 0.0
    run_d = 0.0
    run_grad = 0.0

    for step in range(1, args.max_steps + 1):
        batch_raw = next(data_iter)
        batch = _clone_batch(batch_raw)
        seed_step = args.seed + step

        # Common model inputs.
        _, hist, t_hist, condition, _, generated_tail = _prepare_model_inputs(student, batch, seed_step)

        # Alternate updates (Self-Forcing style): train G on selected steps, D on others.
        train_generator = ((step % max(int(args.generator_update_every), 1)) == 0)

        if train_generator:
            # DMD generator loss: 0.5 * ||x - (x - grad)||^2
            xt_tail, _, timesteps = _sample_noisy_tail(student, generated_tail)
            noisy_full = torch.cat([hist, xt_tail], dim=2)

            with torch.no_grad():
                real_flow = teacher.denoise(noisy_full, timesteps, condition).float().detach()
                fake_flow = critic.denoise(noisy_full, timesteps, condition).float().detach()
                normalizer = torch.abs(generated_tail.detach() - real_flow).mean(
                    dim=list(range(1, real_flow.dim())), keepdim=True
                )
                normalizer = torch.clamp(normalizer, min=float(args.normalizer_eps))
                grad = (fake_flow - real_flow) / normalizer
                target = (generated_tail.detach() - grad.detach()).detach()

            loss_g = float(args.dmd_weight) * 0.5 * F.mse_loss(generated_tail.double(), target.double(), reduction="mean")
            optimizer_g.zero_grad(set_to_none=True)
            loss_g.backward()
            if args.grad_clip_g > 0:
                torch.nn.utils.clip_grad_norm_(trainable_g, args.grad_clip_g)
            optimizer_g.step()

            run_g += float(loss_g.item())
            run_grad += float(torch.abs(grad).mean().item())

        else:
            # Critic diffusion/flow loss on generated samples.
            with torch.no_grad():
                xt_tail, vt_tail_target, timesteps = _sample_noisy_tail(student, generated_tail.detach())
                noisy_full = torch.cat([hist.detach(), xt_tail], dim=2)
            pred_fake_flow = critic.denoise(noisy_full, timesteps, condition).float()
            loss_d = float(args.critic_weight) * F.mse_loss(pred_fake_flow, vt_tail_target.float(), reduction="mean")
            optimizer_d.zero_grad(set_to_none=True)
            loss_d.backward()
            if args.grad_clip_d > 0:
                torch.nn.utils.clip_grad_norm_(trainable_d, args.grad_clip_d)
            optimizer_d.step()
            run_d += float(loss_d.item())

        if step % args.log_every == 0:
            denom = float(args.log_every)
            log_g = run_g / denom
            log_d = run_d / denom
            log_grad = run_grad / denom
            log.info(
                f"[sf_dmd] step={step}/{args.max_steps} "
                f"loss_g={log_g:.6f} "
                f"loss_d={log_d:.6f} "
                f"|grad|={log_grad:.6f}"
            )
            with log_file.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "step": step,
                            "loss_g": log_g,
                            "loss_d": log_d,
                            "grad_abs_mean": log_grad,
                            "train_generator": bool(train_generator),
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
            run_g = 0.0
            run_d = 0.0
            run_grad = 0.0

        if step % args.save_every == 0 or step == args.max_steps:
            ckpt_path = out_dir / f"distill_sf_dmd_step_{step:07d}.pth"
            torch.save(
                {
                    "step": step,
                    "student": student.state_dict(),
                    "critic": critic.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )
            log.info(f"Saved checkpoint to {ckpt_path}")

    log.info("Self-Forcing style DMD distillation finished.")


if __name__ == "__main__":
    main()

