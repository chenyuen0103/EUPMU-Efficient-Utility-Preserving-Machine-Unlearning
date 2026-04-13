#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCH="${ARCH:-resnet18}"
DATASET="${DATASET:-cifar10}"
CLASS_TO_REPLACE="${CLASS_TO_REPLACE:-0}"
FORGET_TAG="${FORGET_TAG:-forget_10.0%}"
MASK="${MASK:-pretrained_models/resnet18/cifar10/model_SA_best.pth.tar}"
SAVE_DIR="${SAVE_DIR:-output_omd_sweep}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-1 2 3 4 5}"
FIXED_TRAIN_SEED="${FIXED_TRAIN_SEED:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-5000}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-8}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-2}"
POLL_SECONDS="${POLL_SECONDS:-10}"
GPU_ALLOWLIST="${GPU_ALLOWLIST:-}"

# OMD-TCH defaults in this repo (no eta sweep by default).
OMD_METHODS="${OMD_METHODS:-omd_tch_eg omd_tch_pgd}"
OMD_ETAS_EG="${OMD_ETAS_EG:-0.1}"
OMD_ETAS_PGD="${OMD_ETAS_PGD:-0.1}"
OMD_RETAIN_WEIGHTS="${OMD_RETAIN_WEIGHTS:-1.0}"
OMD_FORGET_WEIGHTS="${OMD_FORGET_WEIGHTS:-1.0}"

# Generic training args are intentionally fixed here.
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-50}"
UNLEARN_LR="${UNLEARN_LR:-1e-3}"
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

alias_to_canonical_method() {
  local method="$1"
  case "$method" in
    omd_tch|afleg) echo "omd_tch_eg" ;;
    afl) echo "omd_tch_pgd" ;;
    ada_afleg) echo "ada_omd_tch_eg" ;;
    *) echo "$method" ;;
  esac
}

eta_tag() {
  local eta="$1"
  eta="${eta//-/_neg_}"
  eta="${eta//./p}"
  echo "$eta"
}

weight_tag() {
  local weight="$1"
  weight="${weight//-/_neg_}"
  weight="${weight//./p}"
  echo "$weight"
}

run_with_log() {
  local result_path="$1"
  local gpu="$2"
  shift
  shift

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
read -r -a METHOD_LIST <<< "$OMD_METHODS"
read -r -a RETAIN_WEIGHT_LIST <<< "$OMD_RETAIN_WEIGHTS"
read -r -a FORGET_WEIGHT_LIST <<< "$OMD_FORGET_WEIGHTS"

echo "OMD-TCH-specific hyperparameters in this repo:"
echo "  method variant: OMDTCH-EG / OMDTCH-PGD / AdaOMDTCH-EG"
echo "  omd_tch_eta: mirror-descent step size"
echo "  omd_tch_retain_weight, omd_tch_forget_weight: per-task loss scaling before OMD updates"
echo
echo "Not swept because the implementation fixes them to the paper setting:"
echo "  omd_tch_retain_ref = 0.0"
echo "  omd_tch_forget_ref = 0.0"
echo "  omd_tch_rho = 0.0"

for seed in "${SEED_LIST[@]}"; do
  for method in "${METHOD_LIST[@]}"; do
    case "$method" in
      ada_omd_tch_eg|ada_afleg)
        echo "[skip] Adaptive OMD methods are disabled in this sweep: $method"
        continue
        ;;
    esac
    canonical_method="$(alias_to_canonical_method "$method")"
    case "$canonical_method" in
      omd_tch_eg)
        read -r -a ETA_LIST <<< "$OMD_ETAS_EG"
        ;;
      omd_tch_pgd)
        read -r -a ETA_LIST <<< "$OMD_ETAS_PGD"
        ;;
      *)
        echo "[skip] Unknown OMD method: $canonical_method"
        continue
        ;;
    esac
    for eta in "${ETA_LIST[@]}"; do
      eta_suffix="$(eta_tag "$eta")"
      for retain_weight in "${RETAIN_WEIGHT_LIST[@]}"; do
        retain_suffix="$(weight_tag "$retain_weight")"
        for forget_weight in "${FORGET_WEIGHT_LIST[@]}"; do
          forget_suffix="$(weight_tag "$forget_weight")"

          run_tag="seed_${seed}_train_${FIXED_TRAIN_SEED}"
          setting_tag="eta_${eta_suffix}_rw_${retain_suffix}_fw_${forget_suffix}_epoch_${UNLEARN_EPOCHS}"
          result_path="${SAVE_DIR}/${ARCH}/${DATASET}/${FORGET_TAG}/RL/${canonical_method}/${run_tag}/${setting_tag}/evaluation_result.json"

          if [[ -f "$result_path" ]]; then
            echo
            echo "[skip] Found existing result: $result_path"
            continue
          fi

          schedule_if_needed "$result_path" \
            "$PYTHON_BIN" -u main_random.py \
            --arch "$ARCH" \
            --dataset "$DATASET" \
            --class_to_replace "$CLASS_TO_REPLACE" \
            --mask "$MASK" \
            --save_dir "$SAVE_DIR" \
            --seed "$seed" \
            --train_seed "$FIXED_TRAIN_SEED" \
            --wandb_entity "${run_tag}/${setting_tag}" \
            --unlearn RL \
            --unlearn_epochs "$UNLEARN_EPOCHS" \
            --unlearn_lr "$UNLEARN_LR" \
            --mtl \
            --mtl_method "$canonical_method" \
            --omd_tch_retain_weight "$retain_weight" \
            --omd_tch_forget_weight "$forget_weight" \
            --omd_tch_retain_ref 0.0 \
            --omd_tch_forget_ref 0.0 \
            --omd_tch_eta "$eta" \
            --omd_tch_rho 0.0
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
  echo "Some OMD-TCH sweep jobs failed:" >&2
  printf '  %s\n' "${FAILURES[@]}" >&2
  exit 1
fi

echo
echo "OMD-TCH sweep completed."
