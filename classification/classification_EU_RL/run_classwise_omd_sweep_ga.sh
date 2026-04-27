#!/usr/bin/env bash
# run_classwise_omd_sweep_ga.sh
#
# Runs OMD-TCH class-wise unlearning across classes, seeds, and methods with a
# fixed hyperparameter config. Saves results under the naming convention:
#   <SAVE_DIR>/<ARCH>/<DATASET>/<FORGET_TAG>/<UNLEARN_TAG>/<METHOD>/
#       class_<CLS>/seed_<N>_train_<FIXED_TRAIN_SEED>/
#       ulr_<LR>_eta_<E>_rw_<R>_fw_<F>_rho_<RHO>_epoch_<EP>_flt_<TYPE>/
#
# Usage:
#   bash run_classwise_omd_sweep_ga.sh
#
# Key overrides (env vars):
#   DATASET          cifar10 (default) or cifar100
#   CLASSES          space-separated list of class indices to unlearn
#   SEEDS            space-separated seed list (default: 1 2 3 4 5)
#   RUN_NAMESPACE    optional extra path component to isolate a launch
#   SAVE_DIR         output root directory
#   WANDB_PROJECT    set to enable W&B logging (empty = disabled)
#   MAX_PARALLEL_JOBS / MAX_JOBS_PER_GPU   GPU parallelism
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Dataset / model ──────────────────────────────────────────────────────────
ARCH="${ARCH:-resnet18}"
DATASET="${DATASET:-cifar10}"
FORGET_TAG="${FORGET_TAG:-forget_10.0%}"
MASK="${MASK:-pretrained_models/resnet18/cifar10/model_SA_best.pth.tar}"
SAVE_DIR="${SAVE_DIR:-output_classwise_ga}"

# ── Sweep axes ───────────────────────────────────────────────────────────────
# Space-separated list of class indices to unlearn (all 10 CIFAR-10 classes by default).
CLASSES="${CLASSES:-0 1 2 3 4 5 6 7 8 9}"
SEEDS="${SEEDS:-1 2 3 4 5}"
FIXED_TRAIN_SEED="${FIXED_TRAIN_SEED:-1}"

# ── Fixed OMD-TCH config ─────────────────────────────────────────────────────
MTL_METHODS="${MTL_METHODS:-${MTL_METHOD:-omd_tch_eg omd_tch_pgd}}"
OMD_ETA="${OMD_ETA:-0.1}"
OMD_RETAIN_WEIGHT="${OMD_RETAIN_WEIGHT:-1.0}"
OMD_FORGET_WEIGHT="${OMD_FORGET_WEIGHT:-1.0}"
RHO_VALUES="${RHO_VALUES:-0.0 1e-4 1e-2 0.1}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-5}"
UNLEARN_LR="${UNLEARN_LR:-1e-3 1e-2}"
FORGET_LOSS_TYPE="${FORGET_LOSS_TYPE:-ga}"
UNLEARN_TAG="${UNLEARN_TAG:-RL_ga}"

if [[ "$FORGET_LOSS_TYPE" != "ga" ]]; then
  echo "This GA variant expects FORGET_LOSS_TYPE=ga, got: $FORGET_LOSS_TYPE" >&2
  echo "Set FORGET_LOSS_TYPE=ga (or use run_classwise_omd_sweep.sh for random-label mode)." >&2
  exit 1
fi

# ── Parallel job scheduling ───────────────────────────────────────────────────
MIN_FREE_MB="${MIN_FREE_MB:-2500}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-8}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-2}"
POLL_SECONDS="${POLL_SECONDS:-10}"
GPU_ALLOWLIST="${GPU_ALLOWLIST:-}"

# ── Weights & Biases ─────────────────────────────────────────────────────────
WANDB_PROJECT="${WANDB_PROJECT:-}"

# ── Misc ──────────────────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_LOG_NAME="${RUN_LOG_NAME:-run.log}"
RUN_NAMESPACE="${RUN_NAMESPACE:-}"

cd "$SCRIPT_DIR"

# ── Preflight checks ──────────────────────────────────────────────────────────
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; cannot auto-select GPUs." >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if ! "$PYTHON_BIN" - <<'PY'
import importlib, sys
missing = []
for name in ("torch", "torchvision", "numpy", "sklearn", "tqdm", "PIL", "wandb"):
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
  echo "Preflight failed. Fix the Python environment before launching." >&2
  exit 1
fi

# ── Tag helpers ───────────────────────────────────────────────────────────────
# Convert a float like 0.1 → 0p1, or -0.3 → _neg_0p3
float_tag() {
  local v="$1"
  v="${v//-/_neg_}"
  v="${v//./p}"
  echo "$v"
}

# ── GPU scheduling infrastructure ────────────────────────────────────────────
declare -A ALLOWED_GPU=()
if [[ -n "$GPU_ALLOWLIST" ]]; then
  read -r -a GPU_ALLOWLIST_ARRAY <<< "$GPU_ALLOWLIST"
  for gpu in "${GPU_ALLOWLIST_ARRAY[@]}"; do
    ALLOWED_GPU["$gpu"]=1
  done
fi

declare -A GPU_JOB_COUNT=()
declare -A PID_TO_GPU=()
declare -A PID_TO_RESULT=()
declare -a FAILURES=()
active_jobs=0

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

run_with_log() {
  local result_path="$1"
  local gpu="$2"
  shift 2

  local result_dir log_path
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

reap_finished_job() {
  local finished_pid=""
  local exit_code=0

  if wait -n -p finished_pid; then
    exit_code=0
  else
    exit_code=$?
  fi

  [[ -z "$finished_pid" ]] && return 0

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

# ── Build fixed setting tag prefix ───────────────────────────────────────────
eta_suffix="$(float_tag "$OMD_ETA")"
rw_suffix="$(float_tag "$OMD_RETAIN_WEIGHT")"
fw_suffix="$(float_tag "$OMD_FORGET_WEIGHT")"
setting_prefix="eta_${eta_suffix}_rw_${rw_suffix}_fw_${fw_suffix}"

echo "========================================================"
echo "Class-wise OMD-TCH sweep (GA forget loss variant)"
echo "  arch=${ARCH}  dataset=${DATASET}  forget_tag=${FORGET_TAG}"
echo "  classes   : ${CLASSES}"
echo "  seeds     : ${SEEDS}"
echo "  methods   : ${MTL_METHODS}"
echo "  rho sweep : ${RHO_VALUES}"
echo "  unlearn_lr: ${UNLEARN_LR}"
echo "  epochs    : ${UNLEARN_EPOCHS}"
echo "  forget loss type: ${FORGET_LOSS_TYPE}"
echo "  setting   : ulr_<lr>_${setting_prefix}_rho_<rho>_epoch_${UNLEARN_EPOCHS}_flt_${FORGET_LOSS_TYPE}"
echo "  unlearn tag: ${UNLEARN_TAG}"
echo "  save_dir  : ${SAVE_DIR}"
echo "  namespace : ${RUN_NAMESPACE:-<none>}"
echo "  wandb     : ${WANDB_PROJECT:-disabled}"
echo "========================================================"

read -r -a CLASS_LIST <<< "$CLASSES"
read -r -a SEED_LIST  <<< "$SEEDS"
read -r -a RHO_LIST   <<< "$RHO_VALUES"
read -r -a LR_LIST    <<< "$UNLEARN_LR"
read -r -a METHOD_LIST <<< "$MTL_METHODS"

for method in "${METHOD_LIST[@]}"; do
  for cls in "${CLASS_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
      for lr in "${LR_LIST[@]}"; do
        for rho in "${RHO_LIST[@]}"; do
          lr_suffix="$(float_tag "$lr")"
          rho_suffix="$(float_tag "$rho")"
          setting_tag="ulr_${lr_suffix}_${setting_prefix}_rho_${rho_suffix}_epoch_${UNLEARN_EPOCHS}_flt_${FORGET_LOSS_TYPE}"
          class_tag="class_${cls}"
          run_tag="seed_${seed}_train_${FIXED_TRAIN_SEED}"
          wandb_entity_parts=()
          if [[ -n "$RUN_NAMESPACE" ]]; then
            wandb_entity_parts+=("$RUN_NAMESPACE")
          fi
          wandb_entity_parts+=("$class_tag" "$run_tag" "$setting_tag")
          wandb_entity="$(IFS=/; echo "${wandb_entity_parts[*]}")"

          result_dir_parts=("$SAVE_DIR" "$ARCH" "$DATASET" "$FORGET_TAG" "$UNLEARN_TAG" "$method")
          if [[ -n "$RUN_NAMESPACE" ]]; then
            result_dir_parts+=("$RUN_NAMESPACE")
          fi
          result_dir_parts+=("$class_tag" "$run_tag" "$setting_tag")
          result_path="$(IFS=/; echo "${result_dir_parts[*]}")/evaluation_result.json"

          wandb_args=(--wandb_entity "$wandb_entity")
          [[ -n "$WANDB_PROJECT" ]] && wandb_args+=(--wandb_project "$WANDB_PROJECT")

          method_args=(--mtl --mtl_method "$method")
          if [[ "$method" == "chebyshev" ]]; then
            method_args+=(
              --cheby_retain_weight "$OMD_RETAIN_WEIGHT"
              --cheby_forget_weight "$OMD_FORGET_WEIGHT"
              --cheby_retain_ref 0.0
              --cheby_forget_ref -0.3
              --cheby_rho "$rho"
            )
          else
            method_args+=(
              --omd_tch_retain_weight "$OMD_RETAIN_WEIGHT"
              --omd_tch_forget_weight "$OMD_FORGET_WEIGHT"
              --omd_tch_retain_ref 0.0
              --omd_tch_forget_ref -0.3
              --omd_tch_eta "$OMD_ETA"
              --omd_tch_rho "$rho"
            )
          fi

          schedule_if_needed "$result_path" \
            "$PYTHON_BIN" -u main_random.py \
            --arch "$ARCH" \
            --dataset "$DATASET" \
            --class_to_replace "$cls" \
            --mask "$MASK" \
            --save_dir "$SAVE_DIR" \
            --seed "$seed" \
            --train_seed "$FIXED_TRAIN_SEED" \
            "${wandb_args[@]}" \
            --unlearn RL \
            --forget_loss_type "$FORGET_LOSS_TYPE" \
            --unlearn_epochs "$UNLEARN_EPOCHS" \
            --unlearn_lr "$lr" \
            "${method_args[@]}"
        done
      done
    done
  done
done

# ── Wait for all jobs ─────────────────────────────────────────────────────────
while (( active_jobs > 0 )); do
  reap_finished_job
done

if (( ${#FAILURES[@]} > 0 )); then
  echo
  echo "Some jobs failed:" >&2
  printf '  %s\n' "${FAILURES[@]}" >&2
  exit 1
fi

echo
echo "Class-wise OMD-TCH GA sweep completed."
