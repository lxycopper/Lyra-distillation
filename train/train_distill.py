#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-augmentation + DMD distillation training entrypoint for Lyra-2.

This script implements a practical distillation loop tailored for ``Lyra2Model``:
- Teacher: multi-step inference (quality target, no grad).
- Student: DMD 4-step inference path (fast path, with grad on final denoise step).
- Distillation target: generated tail latent chunk (MSE).

Design notes:
- Reuses Lyra-2's existing self-augmentation data preparation by calling
  ``get_data_and_condition`` on the student with ``self_aug_enabled=True``.
- Keeps code independent from unreleased internal training pipelines.
"""

from __future__ import annotations

import argparse
import copy
import importlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from lyra_2._ext.imaginaire.lazy_config import instantiate
from lyra_2._ext.imaginaire.utils import log, misc
from lyra_2._ext.imaginaire.utils.config_helper import get_config_module, override
from lyra_2._src.models.lyra2_model import WAN2PT1_I2V_COND_LATENT_KEY
from lyra_2._src.utils.model_loader import load_model_from_checkpoint


def _clone_batch(x: Any) -> Any:
    """Clone batch structures to avoid in-place mutation collisions."""
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
    """Return negative T5 embeddings, fallback to zeros if missing."""
    neg = data_batch.get("neg_t5_text_embeddings", None)
    pos = data_batch.get("t5_text_embeddings", None)
    if neg is not None:
        return neg
    if pos is None:
        raise KeyError("Missing 't5_text_embeddings' in batch; cannot build text condition for distillation.")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lyra-2 DMD distillation trainer")
    parser.add_argument("--experiment", type=str, default="lyra2")
    parser.add_argument("--config_file", type=str, default="lyra_2/_src/configs/config.py")
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--student_ckpt", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs/distill")
    parser.add_argument("--experiment_opt", action="append", default=[], help="Hydra override, repeatable")

    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-5)
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
    parser.add_argument("--teacher_self_aug", action="store_true", help="Enable teacher self-aug data prep")
    parser.add_argument("--student_self_aug", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    misc.set_random_seed(seed=args.seed, by_rank=False)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataloader from experiment config.
    train_loader, cfg = _build_dataloader(args.config_file, args.experiment, args.experiment_opt)
    data_iter = iter(train_loader)

    # Teacher / student model loading.
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

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student.train()
    # Distillation behavior controls.
    teacher.config.self_aug_enabled = bool(args.teacher_self_aug)
    student.config.self_aug_enabled = bool(args.student_self_aug)

    trainable = _set_trainable_params(student, train_lora_only=bool(args.train_lora_only))
    if len(trainable) == 0:
        raise RuntimeError(
            "No trainable parameters selected. Disable --train_lora_only or load a LoRA-enabled student."
        )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    log.info(
        f"Starting distillation: steps={args.max_steps}, trainable_params={sum(p.numel() for p in trainable):,}"
    )
    running = 0.0

    for step in range(1, args.max_steps + 1):
        batch_raw = next(data_iter)
        batch_teacher = _clone_batch(batch_raw)
        batch_student = _clone_batch(batch_raw)

        # Prepare teacher inputs (clean target path).
        with torch.no_grad():
            _, teacher_x0, _ = teacher.get_data_and_condition(batch_teacher, dropout=False)
            t_hist = int(teacher.framepack_total_max_num_latent_frames - teacher.framepack_num_new_latent_frames)
            teacher_hist = teacher_x0[:, :, :t_hist]
            teacher_cond = batch_teacher[WAN2PT1_I2V_COND_LATENT_KEY]
            teacher_t5 = batch_teacher["t5_text_embeddings"]
            teacher_neg_t5 = _ensure_neg_t5(batch_teacher)
            teacher_tail = teacher.inference(
                history_latents=teacher_hist,
                cond_latent=teacher_cond,
                guidance=float(args.teacher_guidance),
                seed=args.seed + step,
                num_steps=int(args.teacher_steps),
                shift=float(args.teacher_shift),
                t5_text_embeddings=teacher_t5,
                neg_t5_text_embeddings=teacher_neg_t5,
                last_hist_frame=batch_teacher["last_hist_frame"],
                cond_latent_mask=batch_teacher["cond_latent_mask"],
                cond_latent_buffer=batch_teacher.get("cond_latent_buffer", None),
                fps=batch_teacher.get("fps", None),
                padding_mask=batch_teacher.get("padding_mask", None),
            ).detach()

        # Prepare student inputs (self-aug enabled if configured).
        _, student_x0, _ = student.get_data_and_condition(batch_student, dropout=False)
        t_hist_s = int(student.framepack_total_max_num_latent_frames - student.framepack_num_new_latent_frames)
        student_hist = student_x0[:, :, :t_hist_s]
        student_cond = batch_student[WAN2PT1_I2V_COND_LATENT_KEY]
        student_t5 = batch_student["t5_text_embeddings"]
        student_neg_t5 = _ensure_neg_t5(batch_student)

        student_tail = student.inference_dmd(
            history_latents=student_hist,
            cond_latent=student_cond,
            guidance=float(args.student_guidance),
            seed=args.seed + step,
            num_steps=int(args.student_steps),
            shift=float(args.student_shift),
            t5_text_embeddings=student_t5,
            neg_t5_text_embeddings=student_neg_t5,
            last_hist_frame=batch_student["last_hist_frame"],
            cond_latent_mask=batch_student["cond_latent_mask"],
            cond_latent_buffer=batch_student.get("cond_latent_buffer", None),
            fps=batch_student.get("fps", None),
            padding_mask=batch_student.get("padding_mask", None),
        )

        # Core distillation objective on generated tail latents.
        loss_distill = F.mse_loss(student_tail, teacher_tail)
        loss = float(args.distill_weight) * loss_distill

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()

        running += float(loss.item())
        if step % args.log_every == 0:
            avg = running / args.log_every
            running = 0.0
            log.info(
                f"[distill] step={step}/{args.max_steps} "
                f"loss={avg:.6f} distill={float(loss_distill.item()):.6f}"
            )

        if step % args.save_every == 0 or step == args.max_steps:
            ckpt_path = out_dir / f"student_distill_step_{step:07d}.pth"
            torch.save(
                {
                    "step": step,
                    "student": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )
            log.info(f"Saved checkpoint to {ckpt_path}")

    log.info("Distillation finished.")


if __name__ == "__main__":
    main()
