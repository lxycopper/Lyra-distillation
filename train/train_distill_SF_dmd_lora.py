#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-Forcing DMD distillation with **PEFT LoRA** on Lyra-2 under FSDP (multi-GPU, torchrun).

Teacher loads **without** LoRA (matches standard Lyra-2 / DCP base checkpoints). Student and critic
load **with** LoRA injected (rank + targets from CLI); ``framepack_trainable_modules`` is cleared so
only LoRA / adapter weights stay trainable (avoids empty ``--train_lora_only`` filters on KQ names).

Launch::

    torchrun --standalone --nproc_per_node=8 lyra_2/_src/train/train_distill_SF_dmd_lora.py \\
      --teacher_ckpt ... --experiment_opt '+dataloader_train...' ...
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf

from lyra_2._ext.imaginaire.lazy_config import instantiate
from lyra_2._ext.imaginaire.utils import distributed, log, misc
from lyra_2._ext.imaginaire.utils.config_helper import get_config_module, override
from lyra_2._src.models.lyra2_model import WAN2PT1_I2V_COND_LATENT_KEY
from lyra_2._src.modules.conditioner import DataType
from lyra_2._src.utils.model_loader import load_model_from_checkpoint

try:
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover
    from torch.distributed._tensor import DTensor  # type: ignore


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


def _to_cpu_tensors(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _to_cpu_tensors(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_cpu_tensors(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_cpu_tensors(v) for v in x)
    return copy.deepcopy(x)


def _to_device_tensors(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _to_device_tensors(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_device_tensors(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_device_tensors(v, device) for v in x)
    return copy.deepcopy(x)


def _broadcast_batch_from_rank0(batch_on_rank0: dict[str, Any] | None, device: torch.device) -> dict[str, Any]:
    if not dist.is_initialized():
        assert batch_on_rank0 is not None
        return _to_device_tensors(_clone_batch(batch_on_rank0), device)

    payload: list[Any]
    if distributed.is_rank0():
        assert batch_on_rank0 is not None
        payload = [_to_cpu_tensors(_clone_batch(batch_on_rank0))]
    else:
        payload = [None]
    dist.broadcast_object_list(payload, src=0)
    assert payload[0] is not None
    return _to_device_tensors(_clone_batch(payload[0]), device)


def _ensure_neg_t5(data_batch: dict[str, Any]) -> torch.Tensor:
    neg = data_batch.get("neg_t5_text_embeddings", None)
    pos = data_batch.get("t5_text_embeddings", None)
    if neg is not None:
        return neg
    if pos is None:
        raise KeyError("Missing 't5_text_embeddings' in batch.")
    return torch.zeros_like(pos)


def _lora_trainable_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Collect trainable LoRA / PEFT adapter parameters (name heuristic)."""
    out: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nl = name.lower()
        if "lora" in nl or "adapter" in nl:
            out.append(p)
    return out


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

def _cast_float_tensors(x: Any, dtype: torch.dtype) -> Any:
    if torch.is_tensor(x):
        if x.is_floating_point():
            return x.to(dtype=dtype)
        return x
    if isinstance(x, dict):
        return {k: _cast_float_tensors(v, dtype) for k, v in x.items()}
    if isinstance(x, list):
        return [_cast_float_tensors(v, dtype) for v in x]
    if isinstance(x, tuple):
        return tuple(_cast_float_tensors(v, dtype) for v in x)
    return x

def _prepare_model_inputs(model, data_batch: dict[str, Any], seed_step: int, dmd_infer_steps: int = 4):
    # _, x0_latents, _ = model.get_data_and_condition(data_batch, dropout=False)
    # t_hist = int(model.framepack_total_max_num_latent_frames - model.framepack_num_new_latent_frames)
    # hist = x0_latents[:, :, :t_hist]
    # cond_latent = data_batch[WAN2PT1_I2V_COND_LATENT_KEY]
    # cond, uncond = _build_condition_pair(model, data_batch)

    model_dtype = next(model.parameters()).dtype
    data_batch = _cast_float_tensors(data_batch, model_dtype)

    _, x0_latents, _ = model.get_data_and_condition(data_batch, dropout=False)
    t_hist = int(model.framepack_total_max_num_latent_frames - model.framepack_num_new_latent_frames)
    hist = x0_latents[:, :, :t_hist].to(dtype=model_dtype)
    cond_latent = data_batch[WAN2PT1_I2V_COND_LATENT_KEY].to(dtype=model_dtype)

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


def _gather_full_state_dict_cpu(module: torch.nn.Module) -> dict[str, Any]:
    sd = module.state_dict()
    out: dict[str, Any] = {}
    for k, v in sd.items():
        if isinstance(v, DTensor):
            out[k] = v.full_tensor().detach().cpu()
        elif torch.is_tensor(v):
            out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out


def _lora_hydra_overrides(args: argparse.Namespace) -> list[str]:
    """Hydra overrides to enable PEFT LoRA and disable framepack substring trainable list."""
    parts = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    hydra_list = "[" + ",".join(json.dumps(p) for p in parts) + "]"
    opts = [
        "model.config.lora_config.enabled=true",
        f"model.config.lora_config.lora_rank={int(args.lora_rank)}",
        f"model.config.lora_config.lora_target_modules={hydra_list}",
        "model.config.framepack_trainable_modules=",
    ]
    plp = str(args.pretrained_lora_path).strip()
    if plp:
        opts.append(f"model.config.lora_config.pretrained_lora_path={json.dumps(plp)}")
    return opts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Forcing DMD + LoRA (Lyra-2 FSDP, torchrun)")
    parser.add_argument("--train_config", type=str, default="", help="YAML config path for this trainer.")
    parser.add_argument("--experiment", type=str, default="lyra2")
    parser.add_argument("--config_file", type=str, default="lyra_2/_src/configs/config.py")
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--student_ckpt", type=str, default="")
    parser.add_argument("--critic_ckpt", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs/distill_sf_dmd_lora_fsdp")
    parser.add_argument("--experiment_opt", action="append", default=[], help="Hydra override, repeatable")
    parser.add_argument(
        "--fsdp_shard_size",
        type=int,
        default=None,
        help="FSDP sharding group size (defaults to WORLD_SIZE when omitted).",
    )

    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument(
        "--lora_targets",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Comma-separated submodule name fragments for PEFT LoRA (WAN defaults).",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default="",
        help="Optional path to existing LoRA weights (.pth); empty trains from freshly injected adapters.",
    )

    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_g", type=float, default=1.0)
    parser.add_argument("--grad_clip_d", type=float, default=1.0)

    parser.add_argument("--generator_update_every", type=int, default=2)
    parser.add_argument("--dmd_weight", type=float, default=1.0)
    parser.add_argument("--critic_weight", type=float, default=1.0)
    parser.add_argument("--normalizer_eps", type=float, default=1e-6)

    parser.add_argument("--teacher_self_aug", action="store_true")
    parser.add_argument("--student_self_aug", action="store_true", default=True)
    parser.add_argument("--critic_self_aug", action="store_true")
    args = parser.parse_args()

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
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        raise SystemExit(
            "Launch with torchrun (at least 2 processes), e.g.\n"
            "  torchrun --standalone --nproc_per_node=8 lyra_2/_src/train/train_distill_SF_dmd_lora.py \\\n"
            "    --teacher_ckpt ... --experiment_opt '...' ..."
        )

    distributed.init()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    misc.set_random_seed(seed=args.seed, by_rank=True)
    dev = torch.device(f"cuda:{local_rank}")
    log.info(
        f"Distributed CUDA context: rank={distributed.get_rank()} local_rank={local_rank} "
        f"current_device={torch.cuda.current_device()} device_count={torch.cuda.device_count()} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        rank0_only=False,
    )

    fsdp_shard = int(args.fsdp_shard_size) if args.fsdp_shard_size is not None else world_size
    if world_size % fsdp_shard != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must be divisible by fsdp_shard_size={fsdp_shard}")

    experiment_opts_common = list(args.experiment_opt)
    experiment_opts_common.append(f"model.config.fsdp_shard_size={fsdp_shard}")
    experiment_opts_lora = experiment_opts_common + _lora_hydra_overrides(args)

    log.info(f"FSDP shard size={fsdp_shard}, world_size={world_size}, LoRA rank={args.lora_rank}")

    out_dir = Path(args.output_dir)
    if distributed.is_rank0():
        out_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    log_file = out_dir / "train_log.jsonl"

    train_loader, _ = _build_dataloader(args.config_file, args.experiment, experiment_opts_common)
    data_iter = iter(train_loader)

    # Teacher: base Lyra-2 (no LoRA) to match standard checkpoints.
    teacher, _ = load_model_from_checkpoint(
        experiment_name=args.experiment,
        checkpoint_path=args.teacher_ckpt,
        config_file=args.config_file,
        enable_fsdp=True,
        instantiate_ema=False,
        seed=args.seed,
        experiment_opts=experiment_opts_common,
        strict=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student_ckpt = args.student_ckpt if args.student_ckpt else args.teacher_ckpt
    student, _ = load_model_from_checkpoint(
        experiment_name=args.experiment,
        checkpoint_path=student_ckpt,
        config_file=args.config_file,
        enable_fsdp=True,
        instantiate_ema=False,
        seed=args.seed,
        experiment_opts=experiment_opts_lora,
        strict=True,
    )
    student.train()

    critic_ckpt = args.critic_ckpt if args.critic_ckpt else student_ckpt
    critic, _ = load_model_from_checkpoint(
        experiment_name=args.experiment,
        checkpoint_path=critic_ckpt,
        config_file=args.config_file,
        enable_fsdp=True,
        instantiate_ema=False,
        seed=args.seed,
        experiment_opts=experiment_opts_lora,
        strict=True,
    )
    critic.train()

    teacher.config.self_aug_enabled = bool(args.teacher_self_aug)
    student.config.self_aug_enabled = bool(args.student_self_aug)
    critic.config.self_aug_enabled = bool(args.critic_self_aug)

    trainable_g = _lora_trainable_params(student)
    trainable_d = _lora_trainable_params(critic)
    if len(trainable_g) == 0:
        raise RuntimeError(
            "No trainable LoRA parameters on the student. "
            "Check that model.config.lora_config.enabled applied (see logs for PEFT inject) "
            "and that parameter names contain 'lora' or 'adapter'. "
            "Try adjusting --lora_targets to match your backbone (e.g. attn q/k/v/o names)."
        )
    if len(trainable_d) == 0:
        raise RuntimeError("No trainable LoRA parameters on the critic (same checks as student).")

    optimizer_g = torch.optim.AdamW(trainable_g, lr=args.lr_g, weight_decay=args.weight_decay)
    optimizer_d = torch.optim.AdamW(trainable_d, lr=args.lr_d, weight_decay=args.weight_decay)

    log.info(
        "Starting Self-Forcing DMD (FSDP + LoRA): "
        f"steps={args.max_steps}, fsdp_shard={fsdp_shard}, "
        f"trainable_lora_g={sum(p.numel() for p in trainable_g):,}, "
        f"trainable_lora_d={sum(p.numel() for p in trainable_d):,}"
    )

    run_g = 0.0
    run_d = 0.0
    run_grad = 0.0

    for step in range(1, args.max_steps + 1):
        if distributed.is_rank0():
            batch_raw = next(data_iter)
        else:
            batch_raw = None
        batch = _broadcast_batch_from_rank0(batch_raw, dev)
        seed_step = args.seed + step

        _, hist, t_hist, condition, _, generated_tail = _prepare_model_inputs(student, batch, seed_step)

        train_generator = ((step % max(int(args.generator_update_every), 1)) == 0)

        if train_generator:
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
                f"[sf_dmd_lora_fsdp] step={step}/{args.max_steps} "
                f"loss_g={log_g:.6f} loss_d={log_d:.6f} |grad|={log_grad:.6f}"
            )
            if distributed.is_rank0():
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
            student_sd = _gather_full_state_dict_cpu(student)
            critic_sd = _gather_full_state_dict_cpu(critic)
            if distributed.is_rank0():
                ckpt_path = out_dir / f"distill_sf_dmd_lora_fsdp_step_{step:07d}.pth"
                torch.save(
                    {
                        "step": step,
                        "student": student_sd,
                        "critic": critic_sd,
                        "args": vars(args),
                        "note": "FSDP LoRA: full gathered weights; optimizer not saved.",
                    },
                    ckpt_path,
                )
                log.info(f"Saved checkpoint to {ckpt_path}")
            if dist.is_initialized():
                dist.barrier()

    log.info("Self-Forcing DMD distillation (FSDP + LoRA) finished.")


if __name__ == "__main__":
    main()
