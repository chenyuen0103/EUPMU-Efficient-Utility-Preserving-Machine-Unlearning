#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCH="${ARCH:-resnet18}"
DATASET="${DATASET:-cifar10}"
CLASSES="${CLASSES:-0 1 2 3 4 5 6 7 8 9}"
FORGET_TAG="${FORGET_TAG:-forget_10.0%}"
MASK="${MASK:-pretrained_models/resnet18/cifar10/model_SA_best.pth.tar}"
SAVE_DIR="${SAVE_DIR:-output_classwise_ga}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-1 2 3 4 5}"
FIXED_TRAIN_SEED="${FIXED_TRAIN_SEED:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-5000}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-8}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-2}"
POLL_SECONDS="${POLL_SECONDS:-10}"
GPU_ALLOWLIST="${GPU_ALLOWLIST:-}"

# EUPMU-GA settings.
EUPMU_METHODS="${EUPMU_METHODS:-eu eu_fast}"
EU_W_LRS="${EU_W_LRS:-1}"
EU_ERRORS="${EU_ERRORS:-0.01}"
EU_UNLEARN_EPOCHS="${EU_UNLEARN_EPOCHS:-5}"
EU_UNLEARN_LR="${EU_UNLEARN_LR:-1e-3}"
EU_FAST_UNLEARN_LR="${EU_FAST_UNLEARN_LR:-2e-3}"
FORGET_LOSS_TYPE="${FORGET_LOSS_TYPE:-ga}"

if [[ "$FORGET_LOSS_TYPE" != "ga" ]]; then
  echo "This GA variant expects FORGET_LOSS_TYPE=ga, got: $FORGET_LOSS_TYPE" >&2
  echo "Use run_eupmu_sweep.sh for random-label mode." >&2
  exit 1
fi

# Weights & Biases — set WANDB_PROJECT to enable logging (empty = disabled).
WANDB_PROJECT="${WANDB_PROJECT:-eupmu_ga}"
RUN_LOG_NAME="${RUN_LOG_NAME:-run.log}"

cd "$SCRIPT_DIR"

declare -A ALLOWED_GPU=()
if [[ -n "$GPU_ALLOWLIST" ]]; then
  read -r -a GPU_ALLOWLIST_ARRAY <<< "$GPU_ALLOWLIST"
  for gpu in "${GPU_ALLOWLIST_ARRAY[@]}"; do
    ALLOWED_GPU["$gpu"]=1
  done
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if ! "$PYTHON_BIN" - <<'PY'
import importlib
import sys

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
missing = []
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

if int(np.__version__.split(".", 1)[0]) >= 2:
    print(f"NumPy {np.__version__} detected; this repo expects numpy<2.", file=sys.stderr)
    sys.exit(1)
PY
then
  echo "Preflight failed for PYTHON_BIN=$PYTHON_BIN. Fix the Python environment before launching the sweep." >&2
  exit 1
fi

float_tag() {
  local value="$1"
  value="${value//-/_neg_}"
  value="${value//./p}"
  echo "$value"
}

run_with_log() {
  local result_path="$1"
  local gpu="$2"
  shift 2

  local result_dir
  local log_path
  result_dir="$(dirname "$result_path")"
  log_path="$result_dir/$RUN_LOG_NAME"
  mkdir -p "$result_dir"

  echo
  echo "============================================================"
  echo "$*"
  echo "GPU: $gpu"
  echo "Log: $log_path"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="$gpu" "$@" --gpu "$gpu" >"$log_path" 2>&1 &
  local pid=$!

  GPU_JOB_COUNT["$gpu"]=$(( ${GPU_JOB_COUNT["$gpu"]:-0} + 1 ))
  PID_TO_GPU["$pid"]="$gpu"
  PID_TO_RESULT["$pid"]="$result_path"
  ((active_jobs += 1))
}

eligible_gpus() {
  local gpu_index free_mb

  while IFS=, read -r gpu_index free_mb; do
    gpu_index="${gpu_index//[[:space:]]/}"
    free_mb="${free_mb//[[:space:]]/}"

    if [[ -n "$GPU_ALLOWLIST" && -z "${ALLOWED_GPU[$gpu_index]+x}" ]]; then
      continue
    fi
    if (( ${GPU_JOB_COUNT["$gpu_index"]:-0} >= MAX_JOBS_PER_GPU )); then
      continue
    fi
    if (( free_mb > MIN_FREE_MB )); then
      printf '%s\n' "$gpu_index"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
}

reap_finished_job() {
  local finished_pid=""
  local exit_code=0

  if wait -n -p finished_pid; then
    exit_code=0
  else
    exit_code=$?
  fi

  if [[ -z "$finished_pid" ]]; then
    return 0
  fi

  local finished_gpu="${PID_TO_GPU[$finished_pid]:-}"
  local finished_result="${PID_TO_RESULT[$finished_pid]:-}"

  if [[ -n "$finished_gpu" ]]; then
    local remaining=$(( ${GPU_JOB_COUNT["$finished_gpu"]:-0} - 1 ))
    if (( remaining > 0 )); then
      GPU_JOB_COUNT["$finished_gpu"]="$remaining"
    else
      unset 'GPU_JOB_COUNT[$finished_gpu]'
    fi
  fi
  unset 'PID_TO_GPU[$finished_pid]'
  unset 'PID_TO_RESULT[$finished_pid]'
  ((active_jobs -= 1))

  if (( exit_code == 0 )); then
    echo "Finished ${finished_result} on GPU ${finished_gpu}."
  else
    echo "Job ${finished_result} failed on GPU ${finished_gpu} with exit code ${exit_code}." >&2
    FAILURES+=("result=${finished_result} gpu=${finished_gpu} exit_code=${exit_code}")
  fi
}

wait_for_slot() {
  while (( active_jobs >= MAX_PARALLEL_JOBS )); do
    reap_finished_job
  done
}

schedule_if_needed() {
  local result_path="$1"
  shift

  if [[ -f "$result_path" ]]; then
    echo
    echo "[skip] Found existing result: $result_path"
    return 0
  fi

  local gpu
  while true; do
    wait_for_slot
    mapfile -t AVAILABLE_GPUS < <(eligible_gpus)
    if (( ${#AVAILABLE_GPUS[@]} > 0 )); then
      gpu="${AVAILABLE_GPUS[0]}"
      break
    fi
    if (( active_jobs > 0 )); then
      reap_finished_job
    else
      echo
      echo "No eligible GPU found with more than ${MIN_FREE_MB} MB free; retrying in ${POLL_SECONDS}s."
      sleep "$POLL_SECONDS"
    fi
  done

  run_with_log "$result_path" "$gpu" "$@"
}

declare -A GPU_JOB_COUNT=()
declare -A PID_TO_GPU=()
declare -A PID_TO_RESULT=()
declare -a FAILURES=()
active_jobs=0

read -r -a SEED_LIST <<< "$SEEDS"
read -r -a CLASS_LIST <<< "$CLASSES"
read -r -a METHOD_LIST <<< "$EUPMU_METHODS"
read -r -a EU_W_LR_LIST <<< "$EU_W_LRS"
read -r -a EU_ERROR_LIST <<< "$EU_ERRORS"
read -r -a EPOCH_LIST <<< "$EU_UNLEARN_EPOCHS"

echo "EUPMU-GA sweep"
echo "  methods:          ${EUPMU_METHODS}"
echo "  classes:          ${CLASSES}"
echo "  eu_w_lr:          ${EU_W_LRS}"
echo "  eu_error:         ${EU_ERRORS}"
echo "  unlearn_epochs:   ${EU_UNLEARN_EPOCHS}"
echo "  forget_loss_type: ${FORGET_LOSS_TYPE}"
echo "  save_dir:         ${SAVE_DIR}"

for class_to_replace in "${CLASS_LIST[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    for method in "${METHOD_LIST[@]}"; do
      case "$method" in
        eu)
          method_unlearn_lr="$EU_UNLEARN_LR"
          ;;
        eu_fast)
          method_unlearn_lr="$EU_FAST_UNLEARN_LR"
          ;;
        *)
          echo "[skip] Unknown EUPMU method: $method"
          continue
          ;;
      esac

      ulr_suffix="$(float_tag "$method_unlearn_lr")"
      for eu_w_lr in "${EU_W_LR_LIST[@]}"; do
        wlr_suffix="$(float_tag "$eu_w_lr")"
        for eu_error in "${EU_ERROR_LIST[@]}"; do
          err_suffix="$(float_tag "$eu_error")"
          for epochs in "${EPOCH_LIST[@]}"; do
            run_tag="seed_${seed}_train_${FIXED_TRAIN_SEED}"
            class_tag="class_${class_to_replace}"
            setting_tag="ulr_${ulr_suffix}_wlr_${wlr_suffix}_err_${err_suffix}_epoch_${epochs}_flt_${FORGET_LOSS_TYPE}"
            result_path="${SAVE_DIR}/${ARCH}/${DATASET}/${FORGET_TAG}/RL/${method}/${class_tag}/${run_tag}/${setting_tag}/evaluation_result.json"

            if [[ -f "$result_path" ]]; then
              echo
              echo "[skip] Found existing result: $result_path"
              continue
            fi

            wandb_entity="${class_tag}/${run_tag}/${setting_tag}"
            wandb_args=(--wandb_entity "$wandb_entity")
            if [[ -n "$WANDB_PROJECT" ]]; then
              wandb_args+=(--wandb_project "$WANDB_PROJECT")
            fi

            schedule_if_needed "$result_path" \
              "$PYTHON_BIN" -u main_random.py \
              --arch "$ARCH" \
              --dataset "$DATASET" \
              --class_to_replace "$class_to_replace" \
              --mask "$MASK" \
              --save_dir "$SAVE_DIR" \
              --seed "$seed" \
              --train_seed "$FIXED_TRAIN_SEED" \
              "${wandb_args[@]}" \
              --unlearn RL \
              --unlearn_epochs "$epochs" \
              --unlearn_lr "$method_unlearn_lr" \
              --mtl \
              --mtl_method "$method" \
              --eu_w_lr "$eu_w_lr" \
              --eu_error "$eu_error" \
              --forget_loss_type "$FORGET_LOSS_TYPE"
          done
        done
      done
    done
  done
done

while (( active_jobs > 0 )); do
  reap_finished_job
done

if (( ${#FAILURES[@]} > 0 )); then
  echo
  echo "Some EUPMU-GA sweep jobs failed:" >&2
  printf '  %s\n' "${FAILURES[@]}" >&2
  exit 1
fi

echo
echo "EUPMU-GA sweep completed."
