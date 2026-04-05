#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CLASS_LIST="${CLASS_LIST:-0 1 2 3 4 5 6 7 8 9}"
MTL_METHOD="${MTL_METHOD:-eu}"
TRAIN_METHOD="${TRAIN_METHOD:-full}"

TRAIN_ALPHA="${TRAIN_ALPHA:-0.5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
TRAIN_LR="${TRAIN_LR:-1e-5}"
TRAIN_IMAGE_SIZE="${TRAIN_IMAGE_SIZE:-128}"
TRAIN_SEED="${TRAIN_SEED:-}"
DDIM_STEPS="${DDIM_STEPS:-100}"

GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
EVAL_IMAGE_SIZE="${EVAL_IMAGE_SIZE:-512}"
FID_SCRIPT="${FID_SCRIPT:-eval-scripts/compute-fid.py}"
EU_W_LR="${EU_W_LR:-}"
EU_ERROR="${EU_ERROR:-}"
WEIGHT_INIT="${WEIGHT_INIT:-}"

DEVICE_ID="${DEVICE_ID:-0}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
PROMPTS_PATH="${PROMPTS_PATH:-prompts/imagenette.csv}"
CKPT_PATH="${CKPT_PATH:-models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt}"
CONFIG_PATH="${CONFIG_PATH:-configs/stable-diffusion/v1-inference.yaml}"
DIFFUSERS_CONFIG_PATH="${DIFFUSERS_CONFIG_PATH:-diffusers_unet_config.json}"

EVAL_ROOT="${EVAL_ROOT:-evaluation_folder/classwise_pipeline_${MTL_METHOD}}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

RUN_SAVE_REAL_IMAGES="${RUN_SAVE_REAL_IMAGES:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_CONVERT="${RUN_CONVERT:-1}"
RUN_GENERATE="${RUN_GENERATE:-1}"
RUN_FID="${RUN_FID:-1}"
RUN_CLASSIFY="${RUN_CLASSIFY:-1}"
RUN_COLLECT="${RUN_COLLECT:-1}"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "missing required file: $path"
}

setup_runtime_env() {
  local nvidia_lib_dirs=""
  local cupti_dir="/usr/local/cuda-12.0/extras/CUPTI/lib64"

  # Training/generation/eval in this repo currently rely on some packages that may
  # still be resolved from the user site. Keep user-site enabled by default and
  # opt out only for diffusers conversion, where local packages caused import
  # conflicts in this environment.
  unset PYTHONNOUSERSITE || true

  if [[ -d "$HOME/.local/lib/python3.8/site-packages/nvidia" ]]; then
    nvidia_lib_dirs="$(find "$HOME/.local/lib/python3.8/site-packages/nvidia" -type d -name lib | paste -sd:)"
  fi

  if [[ -n "$nvidia_lib_dirs" && -d "$cupti_dir" ]]; then
    export LD_LIBRARY_PATH="${nvidia_lib_dirs}:${cupti_dir}:${LD_LIBRARY_PATH:-}"
  elif [[ -n "$nvidia_lib_dirs" ]]; then
    export LD_LIBRARY_PATH="${nvidia_lib_dirs}:${LD_LIBRARY_PATH:-}"
  elif [[ -d "$cupti_dir" ]]; then
    export LD_LIBRARY_PATH="${cupti_dir}:${LD_LIBRARY_PATH:-}"
  fi
}

latest_model_dir() {
  local class_id="$1"
  local candidates=()
  local path=""
  local filtered=()
  local expected_alpha="-alpha_${TRAIN_ALPHA}"
  local expected_epochs="-epoch_${TRAIN_EPOCHS}"
  local expected_lr="-lr_${TRAIN_LR}"
  local expected_seed=""
  local expected_eu_w_lr=""
  local expected_eu_error=""

  for path in models/compvis-cl-class_"${class_id}"-method_"${TRAIN_METHOD}"*"-mtl_${MTL_METHOD}"*; do
    [[ -d "$path" ]] && candidates+=("$path")
  done

  if [[ -n "$TRAIN_SEED" ]]; then
    expected_seed="-seed_${TRAIN_SEED}"
  fi

  if [[ "$MTL_METHOD" == "eu" ]]; then
    expected_eu_w_lr="-w_lr_${EU_W_LR:-3.0}"
    expected_eu_error="-err_${EU_ERROR:-0.0}"
  fi

  for path in "${candidates[@]}"; do
    [[ "$path" == *"$expected_alpha"* ]] || continue
    [[ "$path" == *"$expected_epochs"* ]] || continue
    [[ "$path" == *"$expected_lr"* ]] || continue
    if [[ -n "$expected_seed" ]]; then
      [[ "$path" == *"$expected_seed"* ]] || continue
    fi
    if [[ -n "$expected_eu_w_lr" ]]; then
      [[ "$path" == *"$expected_eu_w_lr"* ]] || continue
    fi
    if [[ -n "$expected_eu_error" ]]; then
      [[ "$path" == *"$expected_eu_error"* ]] || continue
    fi
    filtered+=("$path")
  done

  candidates=("${filtered[@]}")

  (( ${#candidates[@]} > 0 )) || return 1
  ls -td "${candidates[@]}" | head -n1
}

raw_checkpoint_path() {
  local model_dir="$1"
  local model_name
  model_name="$(basename "$model_dir")"
  printf '%s/%s.pt' "$model_dir" "$model_name"
}

converted_checkpoint_path() {
  local model_dir="$1"
  local model_name
  model_name="$(basename "$model_dir")"
  printf '%s/%s.pt' "$model_dir" "${model_name/compvis/diffusers}"
}

ensure_reference_images() {
  local class_id="$1"
  local ref_dir="imagenette_without_label_${class_id}"

  if [[ "$SKIP_EXISTING" == "1" && -d "$ref_dir" ]]; then
    log "reference set exists for class ${class_id}: ${ref_dir}"
    return
  fi

  log "building reference set for class ${class_id}"
  "$PYTHON_BIN" eval-scripts/save_base_dataset.py --dataset imagenette --label_to_forget "$class_id"
}

train_class() {
  local class_id="$1"
  local existing_dir=""
  local train_args=()

  if existing_dir="$(latest_model_dir "$class_id" 2>/dev/null)"; then
    if [[ "$SKIP_EXISTING" == "1" && -f "$(raw_checkpoint_path "$existing_dir")" ]]; then
      log "raw checkpoint exists for class ${class_id}: $(raw_checkpoint_path "$existing_dir")"
      return
    fi
  fi

  log "training class ${class_id}"
  train_args=(
    train-scripts/random_label_eu.py
    --train_method "$TRAIN_METHOD"
    --alpha "$TRAIN_ALPHA"
    --lr "$TRAIN_LR"
    --epochs "$TRAIN_EPOCHS"
    --class_to_forget "$class_id"
    --device "$DEVICE_ID"
    --mtl
    --mtl_method "$MTL_METHOD"
    --batch_size "$TRAIN_BATCH_SIZE"
    --image_size "$TRAIN_IMAGE_SIZE"
    --ddim_steps "$DDIM_STEPS"
    --ckpt_path "$CKPT_PATH"
    --config_path "$CONFIG_PATH"
    --diffusers_config_path "$DIFFUSERS_CONFIG_PATH"
  )

  if [[ -n "$TRAIN_SEED" ]]; then
    train_args+=(--seed "$TRAIN_SEED")
  fi

  if [[ "$MTL_METHOD" == "eu" ]]; then
    if [[ -n "$EU_W_LR" ]]; then
      train_args+=(--eu_w_lr "$EU_W_LR")
    fi
    if [[ -n "$EU_ERROR" ]]; then
      train_args+=(--eu_error "$EU_ERROR")
    fi
    if [[ -n "$WEIGHT_INIT" ]]; then
      train_args+=(--weight_init "$WEIGHT_INIT")
    fi
  fi

  "$PYTHON_BIN" "${train_args[@]}"
}

convert_class() {
  local class_id="$1"
  local model_dir=""
  local model_name=""
  local converted_path=""

  model_dir="$(latest_model_dir "$class_id")" || die "no trained model directory found for class ${class_id}"
  model_name="$(basename "$model_dir")"
  converted_path="$(converted_checkpoint_path "$model_dir")"

  if [[ "$SKIP_EXISTING" == "1" && -f "$converted_path" ]]; then
    log "converted checkpoint exists for class ${class_id}: ${converted_path}"
    return
  fi

  log "converting checkpoint for class ${class_id}: ${model_name}"
  PYTHONNOUSERSITE=1 PYTHONPATH=train-scripts "$PYTHON_BIN" -c \
    "from convertModels import savemodelDiffusers; savemodelDiffusers('${model_name}', '${CONFIG_PATH}', '${DIFFUSERS_CONFIG_PATH}', device='cpu')"
}

generate_class() {
  local class_id="$1"
  local model_dir=""
  local model_name=""
  local method_root=""
  local eval_dir=""

  model_dir="$(latest_model_dir "$class_id")" || die "no trained model directory found for class ${class_id}"
  model_name="$(basename "$model_dir")"
  method_root="${EVAL_ROOT}/class_${class_id}/${MTL_METHOD}"
  eval_dir="${method_root}/${model_name}"

  if [[ "$SKIP_EXISTING" == "1" && -d "$eval_dir" ]] && compgen -G "${eval_dir}/*.png" >/dev/null; then
    log "generated images exist for class ${class_id}: ${eval_dir}"
    return
  fi

  mkdir -p "$method_root"
  log "generating images for class ${class_id}: ${model_name}"
  PYTHONNOUSERSITE=1 "$PYTHON_BIN" eval-scripts/generate-images.py \
    --prompts_path "$PROMPTS_PATH" \
    --save_path "$method_root" \
    --model_name "$model_name" \
    --device "$EVAL_DEVICE" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --image_size "$EVAL_IMAGE_SIZE" \
    --num_samples "$NUM_SAMPLES" \
    --ddim_steps "$DDIM_STEPS"
}

compute_fid_for_class() {
  local class_id="$1"
  local model_dir=""
  local model_name=""
  local eval_dir=""
  local fid_path=""

  model_dir="$(latest_model_dir "$class_id")" || die "no trained model directory found for class ${class_id}"
  model_name="$(basename "$model_dir")"
  eval_dir="${EVAL_ROOT}/class_${class_id}/${MTL_METHOD}/${model_name}"
  fid_path="${eval_dir}/fid_result.txt"

  [[ -d "$eval_dir" ]] || die "missing generated image folder for class ${class_id}: ${eval_dir}"

  if [[ "$SKIP_EXISTING" == "1" && -f "$fid_path" ]]; then
    log "FID exists for class ${class_id}: ${fid_path}"
    return
  fi

  log "computing FID for class ${class_id}"
  "$PYTHON_BIN" "$FID_SCRIPT" \
    --folder_path "$eval_dir" \
    --class_to_forget "$class_id" \
    --image_size "$EVAL_IMAGE_SIZE" \
    --save_path "$fid_path"
}

classify_class() {
  local class_id="$1"
  local model_dir=""
  local model_name=""
  local eval_dir=""
  local csv_path=""

  model_dir="$(latest_model_dir "$class_id")" || die "no trained model directory found for class ${class_id}"
  model_name="$(basename "$model_dir")"
  eval_dir="${EVAL_ROOT}/class_${class_id}/${MTL_METHOD}/${model_name}"
  csv_path="${eval_dir}/${model_name}_classification.csv"

  [[ -d "$eval_dir" ]] || die "missing generated image folder for class ${class_id}: ${eval_dir}"

  if [[ "$SKIP_EXISTING" == "1" && -f "$csv_path" ]]; then
    log "classification csv exists for class ${class_id}: ${csv_path}"
    return
  fi

  log "classifying generated images for class ${class_id}"
  "$PYTHON_BIN" eval-scripts/imageclassify.py \
    --prompts_path "$PROMPTS_PATH" \
    --folder_path "$eval_dir" \
    --save_path "$csv_path" \
    --device "$EVAL_DEVICE"
}

collect_results() {
  local csv_out="${EVAL_ROOT}/table_${MTL_METHOD}.csv"
  local md_out="${EVAL_ROOT}/table_${MTL_METHOD}.md"

  log "collecting aggregate results"
  "$PYTHON_BIN" eval-scripts/collect-table1-results.py \
    --evaluation_root "$EVAL_ROOT" \
    --prompts_path "$PROMPTS_PATH" \
    --methods "$MTL_METHOD" \
    --class_ids "$CLASS_LIST" \
    --output_csv "$csv_out" \
    --output_md "$md_out"
}

require_file "$PROMPTS_PATH"
require_file "$CKPT_PATH"
require_file "$CONFIG_PATH"
require_file "$DIFFUSERS_CONFIG_PATH"

setup_runtime_env

for class_id in $CLASS_LIST; do
  log "starting pipeline for class ${class_id}"

  if [[ "$RUN_SAVE_REAL_IMAGES" == "1" ]]; then
    ensure_reference_images "$class_id"
  fi

  if [[ "$RUN_TRAIN" == "1" ]]; then
    train_class "$class_id"
  fi

  if [[ "$RUN_CONVERT" == "1" ]]; then
    convert_class "$class_id"
  fi

  if [[ "$RUN_GENERATE" == "1" ]]; then
    generate_class "$class_id"
  fi

  if [[ "$RUN_FID" == "1" ]]; then
    compute_fid_for_class "$class_id"
  fi

  if [[ "$RUN_CLASSIFY" == "1" ]]; then
    classify_class "$class_id"
  fi

  log "finished pipeline for class ${class_id}"
done

if [[ "$RUN_COLLECT" == "1" ]]; then
  collect_results
fi

log "all requested pipeline stages completed"
