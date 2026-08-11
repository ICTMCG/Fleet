#!/usr/bin/env bash
set -euo pipefail

# Batch few-shot: loop over several (checkpoint, prototype) pairs and dataset subsets,
# calling `python -m fleet.train.fewshot_experiment`.
#
# Before use:
#   1) export PYTHONPATH, or run `pip install -e ".[viz]"` locally
#   2) Configure data and weight paths via env vars (see the repo .env.example)
#   3) DINOv3: DINOV3_PATH is optional; if unset, the Python side defaults to <repo>/weights/dinov3-vitl16-pretrain-lvd1689m

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON:-python}"

# Optional: if using Conda, set CONDA_SH and ENV_NAME
CONDA_SH="${CONDA_SH:-}"
ENV_NAME="${ENV_NAME:-}"
if [ -n "$CONDA_SH" ] && [ -f "$CONDA_SH" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  if [ -n "$ENV_NAME" ]; then
    conda activate "$ENV_NAME"
  fi
fi

SKIP_IF_DONE="${SKIP_IF_DONE:-1}"
FORCE="${FORCE:-0}"
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --no-skip) SKIP_IF_DONE=0 ;;
  esac
done

WEIGHTS_CSV="${WEIGHTS_CSV:-128}"
WEIGHT_RUNS_CSV="${WEIGHT_RUNS_CSV:-}"

CKPT_TMPL="${CKPT_TMPL:-${FLEET_PRETRAIN_CHECKPOINT_TMPL:-}}"
PROTO_TMPL="${PROTO_TMPL:-${FLEET_PROTOTYPE_DIR_TMPL:-}}"

PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-${FLEET_PRETRAIN_CHECKPOINT:-}}"
PROTOTYPE_DIR="${PROTOTYPE_DIR:-${FLEET_PROTOTYPE_DIR:-}}"

FAKE_BASE_DIR="${FAKE_BASE_DIR:-${FLEET_OTHER_FAKE_BASE_DIR:-}}"
REAL_SUPPORT_DIR="${REAL_SUPPORT_DIR:-${FLEET_REAL_SUPPORT_DIR:-}}"
REAL_VAL_DIR="${REAL_VAL_DIR:-${FLEET_REAL_VAL_DIR:-}}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-./outputs/fewshot_runs}"
AIGIBENCH_TRAIN="${AIGIBENCH_TRAIN:-${FLEET_AIGIBENCH_TRAIN:-}}"
AIGIBENCH_VAL="${AIGIBENCH_VAL:-${FLEET_AIGIBENCH_VAL:-}}"
DINOV3_PATH="${DINOV3_PATH:-${FLEET_DINOV3_MODEL_PATH:-}}"

SUBSETS_CSV="${SUBSETS_CSV:-ADM,NextStep,Qwen-Image,HunyuanImage-3.0,GPT4O_Image_T2I,FLUX.2,StarGAN,wan2.5-t2i-preview}"
IFS=',' read -r -a DATASETS <<< "$SUBSETS_CSV"

NUM_EPOCHS="${NUM_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
DISTILL_WEIGHT="${DISTILL_WEIGHT:-1.0}"
AVOID_WEIGHT="${AVOID_WEIGHT:-20.0}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
N_FAKE_SUPPORT="${N_FAKE_SUPPORT:-10}"
N_REAL_SUPPORT="${N_REAL_SUPPORT:-10}"

if [ -z "$FAKE_BASE_DIR" ] || [ -z "$REAL_VAL_DIR" ]; then
  echo "Error: please set FAKE_BASE_DIR / REAL_VAL_DIR or the corresponding FLEET_* env vars"
  exit 2
fi
if [ -z "$AIGIBENCH_TRAIN" ] || [ -z "$AIGIBENCH_VAL" ]; then
  echo "Error: please set AIGIBENCH_TRAIN / AIGIBENCH_VAL or FLEET_AIGIBENCH_TRAIN / FLEET_AIGIBENCH_VAL"
  exit 2
fi

echo "[Few-shot batch] root=$REPO_ROOT fake_base=$FAKE_BASE_DIR weights=$WEIGHTS_CSV subsets=$SUBSETS_CSV"

mkdir -p "$OUTPUT_BASE_DIR"

run_one () {
  local ckpt="$1"
  local proto="$2"
  local dataset="$3"
  local tag="$4"

  local query_dir="$FAKE_BASE_DIR/$dataset"
  local out_dir="$OUTPUT_BASE_DIR/$tag/$dataset"
  local done_file="$out_dir/fewshot_results.json"

  if [ ! -d "$query_dir" ]; then
    echo "Warning: dataset directory does not exist: $query_dir"
    return 0
  fi
  if [ ! -f "$ckpt" ]; then
    echo "Error: checkpoint does not exist: $ckpt"
    return 1
  fi
  if [ ! -d "$proto" ]; then
    echo "Error: prototype_dir does not exist: $proto"
    return 1
  fi

  mkdir -p "$out_dir"

  if [ "$SKIP_IF_DONE" = "1" ] && [ "$FORCE" != "1" ] && [ -s "$done_file" ]; then
    echo "[SKIP] weight=$tag dataset=$dataset -> $done_file"
    return 0
  fi

  extra_dino=()
  if [ -n "$DINOV3_PATH" ]; then
    extra_dino=(--dinov3_model_path "$DINOV3_PATH")
  fi

  "$PYTHON_BIN" -m fleet.train.fewshot_experiment \
    --pretrain_checkpoint "$ckpt" \
    --prototype_dir "$proto" \
    --query_dir "$query_dir" \
    --dataset_name "$dataset" \
    --output_dir "$out_dir" \
    "${extra_dino[@]}" \
    --real_val_dir "$REAL_VAL_DIR" \
    --aigibench_train_dir "$AIGIBENCH_TRAIN" \
    --aigibench_val_dir "$AIGIBENCH_VAL" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --distill_weight "$DISTILL_WEIGHT" \
    --num_workers "$NUM_WORKERS" \
    --val_batch_size "$VAL_BATCH_SIZE" \
    --n_fake_support "$N_FAKE_SUPPORT" \
    --n_real_support "$N_REAL_SUPPORT" \
    ${FORCE_TRAIN:+--force_train} \
    ${SAVE_FEWSHOT_MODEL:+--save_fewshot_model}

  echo "[DONE] $tag / $dataset"
}

declare -a RUN_TAGS=()
declare -a RUN_CKPTS=()
declare -a RUN_PROTOS=()

if [ -n "$WEIGHT_RUNS_CSV" ]; then
  IFS=',' read -r -a RUNS <<< "$WEIGHT_RUNS_CSV"
  for item in "${RUNS[@]}"; do
    IFS=':' read -r tag ckpt proto <<< "$item"
    if [ -z "${tag:-}" ] || [ -z "${ckpt:-}" ] || [ -z "${proto:-}" ]; then
      echo "Error: WEIGHT_RUNS_CSV item format should be tag:/abs/ckpt.pth:/abs/proto_dir -> $item"
      exit 2
    fi
    RUN_TAGS+=("$tag")
    RUN_CKPTS+=("$ckpt")
    RUN_PROTOS+=("$proto")
  done
elif [ -n "$PRETRAIN_CHECKPOINT" ] && [ -n "$PROTOTYPE_DIR" ]; then
  RUN_TAGS+=("custom")
  RUN_CKPTS+=("$PRETRAIN_CHECKPOINT")
  RUN_PROTOS+=("$PROTOTYPE_DIR")
else
  if [ -z "$CKPT_TMPL" ] || [ -z "$PROTO_TMPL" ]; then
    echo "Error: WEIGHT_RUNS_CSV or (PRETRAIN_CHECKPOINT+PROTOTYPE_DIR) not set, and CKPT_TMPL/PROTO_TMPL are missing"
    echo "Example: export CKPT_TMPL='/path/pretrain_q%s.pth' PROTO_TMPL='/path/proto_q%s'"
    exit 2
  fi
  IFS=',' read -r -a QS <<< "$WEIGHTS_CSV"
  for q in "${QS[@]}"; do
    q="$(echo "$q" | tr -d ' ')"
    [ -z "$q" ] && continue
    tag="q$q"
    ckpt="$(printf "$CKPT_TMPL" "$q")"
    proto="$(printf "$PROTO_TMPL" "$q")"
    RUN_TAGS+=("$tag")
    RUN_CKPTS+=("$ckpt")
    RUN_PROTOS+=("$proto")
  done
fi

for i in "${!RUN_TAGS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    run_one "${RUN_CKPTS[$i]}" "${RUN_PROTOS[$i]}" "$dataset" "${RUN_TAGS[$i]}"
  done
done

echo "[Few-shot batch] All done, results dir: $OUTPUT_BASE_DIR"
