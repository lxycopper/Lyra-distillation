#!/usr/bin/env bash
# FSDP 2-GPU launcher for train_distill_SF_dmd_2gpus.py (torchrun + Lyra fully_shard).
#
# Usage:
#   export TEACHER_CKPT=/path/to/lyra2/teacher/checkpoint
#   CUDA_VISIBLE_DEVICES=0,1 ./run_2gpus.sh
#
# If CUDA_VISIBLE_DEVICES lists fewer GPUs than NPROC_PER_NODE, ranks share one GPU → OOM on first GPU.
#
# Optional env:
#   NPROC_PER_NODE   (default 2)     — torchrun local processes / visible GPUs used
#   FSDP_SHARD_SIZE  (default unset) — passed as --fsdp_shard_size; omit to use WORLD_SIZE
#   MASTER_PORT      (default 29501) — avoid clash with other torchrun jobs
#   LYRA_SAMPLES, OUTPUT_DIR, MAX_STEPS, STUDENT_CKPT, CRITIC_CKPT, SEED, etc.
#   TRAIN_LORA_ONLY (default 0) — set to 1 only for PEFT LoRA checkpoints (names contain lora/adapter).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LYRA2_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${LYRA2_ROOT}"

export PYTHONPATH="${LYRA2_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Optional: may reduce fragmentation when VRAM is tight.
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

# if [[ -z "${TEACHER_CKPT:-}" ]]; then
#   echo "ERROR: set TEACHER_CKPT to your Lyra-2 teacher checkpoint path (directory or .pth)." >&2
#   exit 1
# fi

OUTPUT_DIR="${OUTPUT_DIR:-${LYRA2_ROOT}/outputs/distill_sf_dmd_fsdp_2gpu}"
MAX_STEPS="${MAX_STEPS:-3000}"
LOG_EVERY="${LOG_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-200}"
SEED="${SEED:-42}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29501}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  _csv="${CUDA_VISIBLE_DEVICES// /}"
  IFS=',' read -ra _cdevs <<< "${_csv}"
  _nvis=0
  for _c in "${_cdevs[@]}"; do
    [[ -n "${_c}" ]] && _nvis=$((_nvis + 1))
  done
  if [[ "${_nvis}" -lt "${NPROC_PER_NODE}" ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES has ${_nvis} entr(y/ies) (${CUDA_VISIBLE_DEVICES}) but NPROC_PER_NODE=${NPROC_PER_NODE}." >&2
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

TRAIN_LORA_FLAG=()
if [[ "${TRAIN_LORA_ONLY:-0}" == "1" ]]; then
  TRAIN_LORA_FLAG=(--train_lora_only)
fi

FSDP_ARGS=()
if [[ -n "${FSDP_SHARD_SIZE:-}" ]]; then
  FSDP_ARGS+=(--fsdp_shard_size "${FSDP_SHARD_SIZE}")
fi

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${SCRIPT_DIR}/train_distill_SF_dmd_2gpus.py" \
  --experiment lyra2 \
  --config_file lyra_2/_src/configs/config.py \
  --teacher_ckpt /home/yudanni/ssd/xyliu/lyra/lyra/Lyra-2/checkpoints/model \
  "${STUDENT_ARGS[@]}" \
  "${CRITIC_ARGS[@]}" \
  "${FSDP_ARGS[@]}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --log_every "${LOG_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --seed "${SEED}" \
  --lr_g "${LR_G:-1e-5}" \
  --lr_d "${LR_D:-1e-5}" \
  --generator_update_every "${GENERATOR_UPDATE_EVERY:-2}" \
  "${TRAIN_LORA_FLAG[@]}" \
  --experiment_opt "+dataloader_train.dataloaders.dl3dv_long_moge_chunk_81_480p_dav3_hsg.dataloader.dataset.dataset_cfg.params.root_path=${DATA_ROOT}" \
  --experiment_opt "+dataloader_train.dataloaders.dl3dv_long_moge_chunk_81_480p_dav3_hsg.dataloader.dataset.dataset_cfg.params.filter_list_path=" \
  --experiment_opt "+dataloader_train.dataloaders.dl3dv_long_moge_chunk_81_480p_dav3_hsg.dataloader.dataset.t5_embedding_path=${T5_EMBED_ROOT}"
