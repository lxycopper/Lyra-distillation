#!/usr/bin/env bash
# 8-GPU FSDP launcher for train_distill_SF_dmd_lora.py (DMD + PEFT LoRA).
#
# Teacher uses the same Lyra-2 config as standard checkpoints (no LoRA).
# Student/critic enable LoRA via Hydra; framepack_trainable_modules is cleared so only LoRA trains.
#
# Usage:
#   export TEACHER_CKPT=/path/to/dcp/parent/or.pth
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_8gpus_lora.sh
#
# Optional env:
#   PRETRAINED_LORA_PATH  — .pth with existing LoRA weights (optional)
#   LORA_RANK             — default 32
#   LORA_TARGETS          — default "q,k,v,o,ffn.0,ffn.2"
#   NPROC_PER_NODE, MASTER_PORT, OUTPUT_DIR, MAX_STEPS, LR_G, LR_D, SEED, LYRA_SAMPLES, etc.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LYRA2_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${LYRA2_ROOT}"

export PYTHONPATH="${LYRA2_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

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

TEACHER_CKPT="${TEACHER_CKPT:-/root/lyra/Lyra-2/checkpoints/model}"

OUTPUT_DIR="${OUTPUT_DIR:-${LYRA2_ROOT}/outputs/distill_sf_dmd_lora_fsdp_8gpu}"
MAX_STEPS="${MAX_STEPS:-3000}"
LOG_EVERY="${LOG_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-200}"
SEED="${SEED:-42}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29509}"
LORA_RANK="${LORA_RANK:-32}"
LORA_TARGETS="${LORA_TARGETS:-q,k,v,o,ffn.0,ffn.2}"

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

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${SCRIPT_DIR}/train_distill_SF_dmd_lora.py" \
  --experiment lyra2 \
  --config_file lyra_2/_src/configs/config.py \
  --teacher_ckpt "${TEACHER_CKPT}" \
  "${STUDENT_ARGS[@]}" \
  "${CRITIC_ARGS[@]}" \
  "${FSDP_ARGS[@]}" \
  "${PRETRAINED_LORA_ARGS[@]}" \
  --lora_rank "${LORA_RANK}" \
  --lora_targets "${LORA_TARGETS}" \
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
