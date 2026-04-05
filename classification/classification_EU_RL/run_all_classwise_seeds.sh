#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS="${SEEDS:-1 2 3 4 5}"
WANDB_ENTITY_TAG_PREFIX="${WANDB_ENTITY_TAG_PREFIX:-seed_}"
MIN_FREE_MB="${MIN_FREE_MB:-10000}"
POLL_SECONDS="${POLL_SECONDS:-30}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
GPU_ALLOWLIST="${GPU_ALLOWLIST:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$SCRIPT_DIR"
mkdir -p "$LOG_DIR"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; cannot auto-select GPUs." >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if ! "$PYTHON_BIN" - <<'PY'
import importlib
import sys

missing = []
required_modules = (
    "torch",
    "torchvision",
    "numpy",
    "sklearn",
    "tqdm",
    "PIL",
    "datasets",
    "lmdb",
    "matplotlib",
    "wandb",
    "six",
)
for name in required_modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")

if missing:
    print("Python environment check failed:", file=sys.stderr)
    for item in missing:
        print(f"  {item}", file=sys.stderr)
    sys.exit(1)

import numpy as np

major = int(np.__version__.split(".", 1)[0])
if major >= 2:
    print(
        f"NumPy {np.__version__} detected; this repo expects numpy<2.",
        file=sys.stderr,
    )
    sys.exit(1)
PY
then
  echo "Preflight failed for PYTHON_BIN=$PYTHON_BIN. Fix the Python environment before launching multi-seed jobs." >&2
  exit 1
fi

read -r -a SEED_LIST <<< "$SEEDS"

declare -A BUSY_GPUS=()
declare -A PID_TO_GPU=()
declare -A PID_TO_SEED=()
declare -A ALLOWED_GPU=()
declare -a FAILURES=()

if [[ -n "$GPU_ALLOWLIST" ]]; then
  read -r -a GPU_ALLOWLIST_ARRAY <<< "$GPU_ALLOWLIST"
  for gpu in "${GPU_ALLOWLIST_ARRAY[@]}"; do
    ALLOWED_GPU["$gpu"]=1
  done
fi

eligible_gpus() {
  local gpu_index free_mb

  while IFS=, read -r gpu_index free_mb; do
    gpu_index="${gpu_index//[[:space:]]/}"
    free_mb="${free_mb//[[:space:]]/}"

    if [[ -n "$GPU_ALLOWLIST" && -z "${ALLOWED_GPU[$gpu_index]+x}" ]]; then
      continue
    fi
    if [[ -n "${BUSY_GPUS[$gpu_index]+x}" ]]; then
      continue
    fi
    if (( free_mb > MIN_FREE_MB )); then
      printf '%s\n' "$gpu_index"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
}

launch_seed() {
  local seed="$1"
  local gpu="$2"
  local train_seed="${TRAIN_SEED:-$seed}"
  local log_path="$LOG_DIR/seed_${seed}_gpu_${gpu}.log"
  local pid

  echo
  echo "############################################################"
  echo "Running class-wise suite for seed ${seed} on GPU ${gpu}"
  echo "Log: ${log_path}"
  echo "############################################################"

  SEED="$seed" \
  TRAIN_SEED="$train_seed" \
  WANDB_ENTITY_TAG="${WANDB_ENTITY_TAG_PREFIX}${seed}" \
  CUDA_VISIBLE_DEVICES="$gpu" \
  GPU="$gpu" \
  LOCAL_GPU="0" \
  bash ./run_all_classwise.sh >"$log_path" 2>&1 &
  pid=$!

  BUSY_GPUS["$gpu"]=1
  PID_TO_GPU["$pid"]="$gpu"
  PID_TO_SEED["$pid"]="$seed"
}

seed_index=0
active_jobs=0
had_failures=0

while (( seed_index < ${#SEED_LIST[@]} || active_jobs > 0 )); do
  while (( seed_index < ${#SEED_LIST[@]} )); do
    mapfile -t AVAILABLE_GPUS < <(eligible_gpus)
    if (( ${#AVAILABLE_GPUS[@]} == 0 )); then
      break
    fi

    launch_seed "${SEED_LIST[$seed_index]}" "${AVAILABLE_GPUS[0]}"
    ((seed_index += 1))
    ((active_jobs += 1))
  done

  if (( seed_index >= ${#SEED_LIST[@]} && active_jobs == 0 )); then
    break
  fi

  if (( active_jobs == 0 )); then
    echo
    echo "No eligible GPU found with more than ${MIN_FREE_MB} MB free; retrying in ${POLL_SECONDS}s."
    sleep "$POLL_SECONDS"
    continue
  fi

  finished_pid=""
  if wait -n -p finished_pid; then
    exit_code=0
  else
    exit_code=$?
  fi

  finished_gpu="${PID_TO_GPU[$finished_pid]}"
  finished_seed="${PID_TO_SEED[$finished_pid]}"

  unset 'BUSY_GPUS[$finished_gpu]'
  unset 'PID_TO_GPU[$finished_pid]'
  unset 'PID_TO_SEED[$finished_pid]'
  ((active_jobs -= 1))

  if (( exit_code == 0 )); then
    echo "Seed ${finished_seed} finished successfully on GPU ${finished_gpu}."
  else
    echo "Seed ${finished_seed} failed on GPU ${finished_gpu} with exit code ${exit_code}." >&2
    FAILURES+=("seed=${finished_seed} gpu=${finished_gpu} exit_code=${exit_code}")
    had_failures=1
  fi
done

if (( had_failures )); then
  echo
  echo "Some runs failed:"
  for failure in "${FAILURES[@]}"; do
    echo "  ${failure}"
  done
  exit 1
fi
