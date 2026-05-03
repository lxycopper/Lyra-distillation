#!/usr/bin/env bash
# DMD (Self-Forcing style) distillation launcher for train_distill_SF_dmd.py
# Usage:
#   export TEACHER_CKPT=/path/to/lyra2/teacher/checkpoint
#   CUDA_VISIBLE_DEVICES=0,1 ./run.sh   # use physical GPUs 0 and 1 (see note below)
# Optional: MAX_STEPS, OUTPUT_DIR, STUDENT_CKPT, CRITIC_CKPT
#
# GPU note: this trainer is single-process; 0,1 restricts which cards are visible.
# Training typically uses one logical GPU (cuda:0 after masking). Multi-GPU needs code/torchrun changes.
# VRAM: SF-DMD loads teacher + student + critic (~3 full Lyra-2 nets). A single ~44GB card is often
# insufficient even for the first load; prefer 80GB+ or free the GPU and try PYTORCH_CUDA_ALLOC_CONF above.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Lyra-2 repo root (directory that contains the lyra_2 package)
LYRA2_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${LYRA2_ROOT}"

export PYTHONPATH="${LYRA2_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Optional: may reduce fragmentation when VRAM is tight (see PyTorch CUDA memory docs).
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Preprocessed chunks under lyra_samples (same layout as distill_sf_dmd_sample8_config.yaml)
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


OUTPUT_DIR="${OUTPUT_DIR:-${LYRA2_ROOT}/outputs/distill_sf_dmd_lyra_samples}"
MAX_STEPS="${MAX_STEPS:-3000}"
LOG_EVERY="${LOG_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-200}"
SEED="${SEED:-42}"

STUDENT_ARGS=()
if [[ -n "${STUDENT_CKPT:-}" ]]; then
  STUDENT_ARGS+=(--student_ckpt "${STUDENT_CKPT}")
fi
CRITIC_ARGS=()
if [[ -n "${CRITIC_CKPT:-}" ]]; then
  CRITIC_ARGS+=(--critic_ckpt "${CRITIC_CKPT}")
fi

TRAIN_LORA_FLAG=()
if [[ "${TRAIN_LORA_ONLY:-1}" == "1" ]]; then
  TRAIN_LORA_FLAG=(--train_lora_only)
fi

exec python "${SCRIPT_DIR}/train_distill_SF_dmd.py" \
  --experiment lyra2 \
  --config_file lyra_2/_src/configs/config.py \
  --teacher_ckpt /root/lyra/Lyra-2/checkpoints/model\
  "${STUDENT_ARGS[@]}" \
  "${CRITIC_ARGS[@]}" \
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
