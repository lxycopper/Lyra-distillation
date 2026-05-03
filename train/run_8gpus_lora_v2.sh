#!/usr/bin/env bash
# 8-GPU launcher for train_distill_SF_dmd_lora_v2.py (tighter peak VRAM vs v1).
#
# v2 defaults: dmd_infer_steps=1, extra empty_cache after student prepare and between
# teacher/critic denoise, LoRA rank=16, num_workers=0, self_aug off (Hydra).
# For fragmentation near OOM: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#
# Usage:
#   export TEACHER_CKPT=/path/to/teacher
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_8gpus_lora_v2.sh
#
# Optional env:
#   DMD_INFER_STEPS  — 1–4, default 1 in v2
#   MAX_INPUT_FRAMES — if set (e.g. 48), passed as --max_input_frames
#   LORA_RANK, EMPTY_CACHE_EVERY, ENABLE_SELF_AUG, LOSS_FP64, PRETRAINED_LORA_PATH, OUTPUT_DIR, etc.
#   NO_EMPTY_CACHE_AFTER_PREPARE=1  — passes --no-empty-cache-after-prepare
#   NO_EMPTY_CACHE_BETWEEN_TEACHER_CRITIC=1 — passes --no-empty-cache-between-teacher-critic

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LYRA2_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${LYRA2_ROOT}"

export PYTHONPATH="${LYRA2_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

LYRA_SAMPLES="${LYRA_SAMPLES:-${SCRIPT_DIR}/lyra_samples}"
DATA_ROOT="${LYRA_SAMPLES}/DL3DV-ALL-480P-lyra-sample8"
T5_EMBED_ROOT="${DATA_ROOT}/t5_embeddings"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "ERROR: dataset root not found: ${DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${T5_EMBED_ROOT}" ]]; then
  echo "ERROR: t5_embeddings dir not found: ${T5_EMBED_ROOT}" >&2
  exit 1
fi

TEACHER_CKPT="${TEACHER_CKPT:-/root/lyra/Lyra-2/checkpoints/model}"

OUTPUT_DIR="${OUTPUT_DIR:-${LYRA2_ROOT}/outputs/distill_sf_dmd_lora_v2_fsdp_8gpu}"
MAX_STEPS="${MAX_STEPS:-3000}"
LOG_EVERY="${LOG_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-200}"
SEED="${SEED:-42}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29511}"
LORA_RANK="${LORA_RANK:-16}"
LORA_TARGETS="${LORA_TARGETS:-q,k,v,o,ffn.0,ffn.2}"
DMD_INFER_STEPS="${DMD_INFER_STEPS:-1}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-1}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  _csv="${CUDA_VISIBLE_DEVICES// /}"
  IFS=',' read -ra _cdevs <<< "${_csv}"
  _nvis=0
  for _c in "${_cdevs[@]}"; do
    [[ -n "${_c}" ]] && _nvis=$((_nvis + 1))
  done
  if [[ "${_nvis}" -lt "${NPROC_PER_NODE}" ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES has ${_nvis} entr(y/ies) but NPROC_PER_NODE=${NPROC_PER_NODE}." >&2
    exit 1
  fi
fi

STUDENT_ARGS=()
if [[ -n "${STUDENT_CKPT:-}" ]]; then
  STUDENT_ARGS+=(--student_ckpt "${STUDENT_CKPT}")
fi
CRITIC_ARGS=()
if [[ -n "${CRITIC_CKPT:-}" ]]; then
  CRITIC_ARGS+=(--critic_ckpt "${CRITIC_CKPT}")
fi

FSDP_ARGS=()
if [[ -n "${FSDP_SHARD_SIZE:-}" ]]; then
  FSDP_ARGS+=(--fsdp_shard_size "${FSDP_SHARD_SIZE}")
fi

PRETRAINED_LORA_ARGS=()
if [[ -n "${PRETRAINED_LORA_PATH:-}" ]]; then
  PRETRAINED_LORA_ARGS+=(--pretrained_lora_path "${PRETRAINED_LORA_PATH}")
fi

MAX_FRAMES_ARGS=()
if [[ -n "${MAX_INPUT_FRAMES:-}" ]]; then
  MAX_FRAMES_ARGS+=(--max_input_frames "${MAX_INPUT_FRAMES}")
fi

EXTRA_V2=()
if [[ "${ENABLE_SELF_AUG:-0}" == "1" ]]; then
  EXTRA_V2+=(--enable_self_aug)
fi
if [[ "${LOSS_FP64:-0}" == "1" ]]; then
  EXTRA_V2+=(--loss_fp64)
fi
if [[ "${NO_EMPTY_CACHE_AFTER_PREPARE:-0}" == "1" ]]; then
  EXTRA_V2+=(--no-empty-cache-after-prepare)
fi
if [[ "${NO_EMPTY_CACHE_BETWEEN_TEACHER_CRITIC:-0}" == "1" ]]; then
  EXTRA_V2+=(--no-empty-cache-between-teacher-critic)
fi

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${SCRIPT_DIR}/train_distill_SF_dmd_lora_v2.py" \
  --experiment lyra2 \
  --config_file lyra_2/_src/configs/config.py \
  --teacher_ckpt "${TEACHER_CKPT}" \
  "${STUDENT_ARGS[@]}" \
  "${CRITIC_ARGS[@]}" \
  "${FSDP_ARGS[@]}" \
  "${PRETRAINED_LORA_ARGS[@]}" \
  "${MAX_FRAMES_ARGS[@]}" \
  --lora_rank "${LORA_RANK}" \
  --lora_targets "${LORA_TARGETS}" \
  --dmd_infer_steps "${DMD_INFER_STEPS}" \
  --empty_cache_every "${EMPTY_CACHE_EVERY}" \
  "${EXTRA_V2[@]}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --log_every "${LOG_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --seed "${SEED}" \
  --lr_g "${LR_G:-1e-4}" \
  --lr_d "${LR_D:-1e-4}" \
  --generator_update_every "${GENERATOR_UPDATE_EVERY:-2}" \
  --experiment_opt "+dataloader_train.dataloaders.dl3dv_long_moge_chunk_81_480p_dav3_hsg.dataloader.dataset.dataset_cfg.params.root_path=${DATA_ROOT}" \
  --experiment_opt "+dataloader_train.dataloaders.dl3dv_long_moge_chunk_81_480p_dav3_hsg.dataloader.dataset.dataset_cfg.params.filter_list_path=" \
  --experiment_opt "+dataloader_train.dataloaders.dl3dv_long_moge_chunk_81_480p_dav3_hsg.dataloader.dataset.t5_embedding_path=${T5_EMBED_ROOT}"


# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export MAX_INPUT_FRAMES=48 
# ./run_8gpus_lora_v2.sh