#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-augmentation + DMD-style distillation training for Lyra-2.

This is a v2 trainer that extends ``train_distill.py`` with a lightweight
distribution-matching branch:

- Teacher (frozen): multi-step inference + reference score model.
- Student (trainable): DMD fast path generator.
- Critic/Fake-score (trainable): approximates teacher score on noisy generated tails.

Losses:
1) Distill loss: ``MSE(student_tail, teacher_tail)``.
2) Generator DMD loss (latent-flow variant): ``0.5 * ||x - (x - grad)||^2`` where
   ``grad = (fake_score - real_score) / normalizer``.
3) Critic loss: ``MSE(fake_score, real_score_detached)`` on noisy generated tails.
"""

from __future__ import annotations

import argparse
import copy
import importlib
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn.functional as F
from lyra_2._ext.imaginaire.lazy_config import instantiate
from lyra_2._ext.imaginaire.utils import log, misc
from lyra_2._ext.imaginaire.utils.config_helper import get_config_module, override
from lyra_2._src.models.lyra2_model import WAN2PT1_I2V_COND_LATENT_KEY
from lyra_2._src.utils.model_loader import load_model_from_checkpoint


def _clone_batch(x: Any) -> Any:
    """Clone nested batch objects to avoid in-place mutation collisions."""
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


def _sample_noisy_tail(model, clean_tail: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample noisy tail and corresponding timesteps via model RectifiedFlow."""
    b = int(clean_tail.shape[0])
    t_b = model.rectified_flow.sample_train_time(b).to(**model.flow_matching_kwargs)
    t_b = t_b.reshape(b, 1)
    timesteps = model.rectified_flow.get_discrete_timestamp(t_b, model.flow_matching_kwargs)
    sigmas = model.rectified_flow.get_sigmas(timesteps, model.flow_matching_kwargs).reshape(b, 1)
    eps = torch.randn_like(clean_tail, dtype=model.flow_matching_kwargs["dtype"])
    noisy_tail, _ = model.rectified_flow.get_interpolation(
        eps.to(dtype=torch.float32), clean_tail.to(dtype=torch.float32), sigmas
    )
    return noisy_tail.to(dtype=clean_tail.dtype), timesteps.reshape(b, 1)


def _predict_tail_flow(model, noisy_full: torch.Tensor, timesteps: torch.Tensor, condition) -> torch.Tensor:
    """Return generated-tail flow prediction from Lyra2 denoiser."""
    return model.denoise(noisy_full, timesteps, condition).float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lyra-2 DMD-style distillation trainer")
    parser.add_argument("--experiment", type=str, default="lyra2")
    parser.add_argument("--config_file", type=str, default="lyra_2/_src/configs/config.py")
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--student_ckpt", type=str, default="")
    parser.add_argument("--critic_ckpt", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs/distill_dmd")
    parser.add_argument("--experiment_opt", action="append", default=[], help="Hydra override, repeatable")

    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_critic", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--train_lora_only", action="store_true")

    parser.add_argument("--teacher_guidance", type=float, default=1.5)
    parser.add_argument("--teacher_steps", type=int, default=35)
    parser.add_argument("--teacher_shift", type=float, default=5.0)

    parser.add_argument("--student_guidance", type=float, default=1.0)
    parser.add_argument("--student_steps", type=int, default=4)
    parser.add_argument("--student_shift", type=float, default=5.0)

    parser.add_argument("--distill_weight", type=float, default=1.0)
    parser.add_argument("--dmd_weight", type=float, default=0.5)
    parser.add_argument("--critic_weight", type=float, default=1.0)
    parser.add_argument("--critic_update_every", type=int, default=1)

    parser.add_argument("--teacher_self_aug", action="store_true", help="Enable teacher self-aug data prep")
    parser.add_argument("--student_self_aug", action="store_true", default=True)
    parser.add_argument("--critic_self_aug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    misc.set_random_seed(seed=args.seed, by_rank=False)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, _ = _build_dataloader(args.config_file, args.experiment, args.experiment_opt)
    data_iter = iter(train_loader)

    # Teacher.
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

    # Student.
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

    # Critic/fake-score model (same architecture; trained as score approximator).
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
        raise RuntimeError("No trainable student parameters selected.")
    if len(trainable_d) == 0:
        raise RuntimeError("No trainable critic parameters selected.")

    optimizer_g = torch.optim.AdamW(trainable_g, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_d = torch.optim.AdamW(trainable_d, lr=args.lr_critic, weight_decay=args.weight_decay)

    log.info(
        "Starting DMD-style distillation: "
        f"steps={args.max_steps}, "
        f"trainable_student={sum(p.numel() for p in trainable_g):,}, "
        f"trainable_critic={sum(p.numel() for p in trainable_d):,}"
    )

    running_total = 0.0
    running_distill = 0.0
    running_dmd = 0.0
    running_critic = 0.0

    for step in range(1, args.max_steps + 1):
        batch_raw = next(data_iter)
        batch_teacher = _clone_batch(batch_raw)
        batch_student = _clone_batch(batch_raw)

        # Teacher target (clean path).
        with torch.no_grad():
            _, teacher_x0, _ = teacher.get_data_and_condition(batch_teacher, dropout=False)
            t_hist = int(teacher.framepack_total_max_num_latent_frames - teacher.framepack_num_new_latent_frames)
            teacher_hist = teacher_x0[:, :, :t_hist]
            teacher_cond = batch_teacher[WAN2PT1_I2V_COND_LATENT_KEY]
            teacher_tail = teacher.inference(
                history_latents=teacher_hist,
                cond_latent=teacher_cond,
                guidance=float(args.teacher_guidance),
                seed=args.seed + step,
                num_steps=int(args.teacher_steps),
                shift=float(args.teacher_shift),
                t5_text_embeddings=batch_teacher["t5_text_embeddings"],
                neg_t5_text_embeddings=_ensure_neg_t5(batch_teacher),
                last_hist_frame=batch_teacher["last_hist_frame"],
                cond_latent_mask=batch_teacher["cond_latent_mask"],
                cond_latent_buffer=batch_teacher.get("cond_latent_buffer", None),
                fps=batch_teacher.get("fps", None),
                padding_mask=batch_teacher.get("padding_mask", None),
            ).detach()

        # Student prep and DMD-path generation.
        _, student_x0, condition_student = student.get_data_and_condition(batch_student, dropout=False)
        t_hist_s = int(student.framepack_total_max_num_latent_frames - student.framepack_num_new_latent_frames)
        student_hist = student_x0[:, :, :t_hist_s]
        student_cond = batch_student[WAN2PT1_I2V_COND_LATENT_KEY]
        student_tail = student.inference_dmd(
            history_latents=student_hist,
            cond_latent=student_cond,
            guidance=float(args.student_guidance),
            seed=args.seed + step,
            num_steps=int(args.student_steps),
            shift=float(args.student_shift),
            t5_text_embeddings=batch_student["t5_text_embeddings"],
            neg_t5_text_embeddings=_ensure_neg_t5(batch_student),
            last_hist_frame=batch_student["last_hist_frame"],
            cond_latent_mask=batch_student["cond_latent_mask"],
            cond_latent_buffer=batch_student.get("cond_latent_buffer", None),
            fps=batch_student.get("fps", None),
            padding_mask=batch_student.get("padding_mask", None),
        )

        # -----------------------
        # Generator update.
        # -----------------------
        loss_distill = F.mse_loss(student_tail, teacher_tail)

        noisy_tail, timesteps = _sample_noisy_tail(student, student_tail)
        noisy_full = torch.cat([student_hist, noisy_tail], dim=2)

        with torch.no_grad():
            # Real score from frozen teacher.
            real_flow = _predict_tail_flow(teacher, noisy_full, timesteps, condition_student).detach()
            # Fake score from critic used as DMD guidance for generator update.
            fake_flow_ng = _predict_tail_flow(critic, noisy_full, timesteps, condition_student).detach()
            normalizer = torch.abs(student_tail.detach() - real_flow).mean(
                dim=list(range(1, real_flow.dim())), keepdim=True
            )
            normalizer = torch.clamp(normalizer, min=1e-6)
            dmd_grad = (fake_flow_ng - real_flow) / normalizer
            dmd_target = (student_tail.detach() - dmd_grad).detach()

        loss_dmd = 0.5 * F.mse_loss(student_tail, dmd_target)
        loss_g = float(args.distill_weight) * loss_distill + float(args.dmd_weight) * loss_dmd

        optimizer_g.zero_grad(set_to_none=True)
        loss_g.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_g, args.grad_clip)
        optimizer_g.step()

        # -----------------------
        # Critic update.
        # -----------------------
        loss_critic = torch.tensor(0.0, device=student_tail.device)
        if (step % max(int(args.critic_update_every), 1)) == 0:
            with torch.no_grad():
                noisy_tail_d, timesteps_d = _sample_noisy_tail(student, student_tail.detach())
                noisy_full_d = torch.cat([student_hist.detach(), noisy_tail_d], dim=2)
                real_flow_d = _predict_tail_flow(teacher, noisy_full_d, timesteps_d, condition_student).detach()
            fake_flow_d = _predict_tail_flow(critic, noisy_full_d, timesteps_d, condition_student)
            loss_critic = float(args.critic_weight) * F.mse_loss(fake_flow_d, real_flow_d)
            optimizer_d.zero_grad(set_to_none=True)
            loss_critic.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_d, args.grad_clip)
            optimizer_d.step()

        running_total += float(loss_g.item())
        running_distill += float(loss_distill.item())
        running_dmd += float(loss_dmd.item())
        running_critic += float(loss_critic.item())

        if step % args.log_every == 0:
            denom = float(args.log_every)
            log.info(
                f"[distill_dmd] step={step}/{args.max_steps} "
                f"loss_g={running_total/denom:.6f} "
                f"distill={running_distill/denom:.6f} "
                f"dmd={running_dmd/denom:.6f} "
                f"critic={running_critic/denom:.6f}"
            )
            running_total = 0.0
            running_distill = 0.0
            running_dmd = 0.0
            running_critic = 0.0

        if step % args.save_every == 0 or step == args.max_steps:
            ckpt_path = out_dir / f"distill_dmd_step_{step:07d}.pth"
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

    log.info("DMD-style distillation finished.")


if __name__ == "__main__":
    main()

