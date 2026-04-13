#!/usr/bin/env bash
# Run all classwise methods except retrain. The suite itself now fans methods
# out across all visible GPUs, so this wrapper just runs one seed at a time.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS="${SEEDS:-1 2 3 4 5}"
FIXED_TRAIN_SEED="${FIXED_TRAIN_SEED:-1}"
SAVE_DIR="${SAVE_DIR:-output}"
WANDB_ENTITY_TAG_PREFIX="${WANDB_ENTITY_TAG_PREFIX:-seed_}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs_no_retrain}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_RETRAIN="${SKIP_RETRAIN:-1}"

cd "$SCRIPT_DIR"
mkdir -p "$LOG_DIR"

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
  echo "Preflight failed for PYTHON_BIN=$PYTHON_BIN. Fix the Python environment before launching multi-seed jobs." >&2
  exit 1
fi

read -r -a SEED_LIST <<< "$SEEDS"

for seed in "${SEED_LIST[@]}"; do
  log_path="$LOG_DIR/seed_${seed}.log"
  echo
  echo "############################################################"
  echo "Running class-wise suite for seed ${seed}"
  echo "Log: ${log_path}"
  echo "############################################################"

  SEED="$seed" \
  TRAIN_SEED="$FIXED_TRAIN_SEED" \
  SAVE_DIR="$SAVE_DIR" \
  WANDB_ENTITY_TAG="${WANDB_ENTITY_TAG_PREFIX}${seed}_train_1" \
  SKIP_RETRAIN="$SKIP_RETRAIN" \
  bash ./run_all_classwise.sh >"$log_path" 2>&1
done

echo
echo "All seeded runs completed successfully!"
