#!/usr/bin/env bash
# Fleet full pipeline: pretraining + few-shot fine-tuning + summary
# Usage: bash Run.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
ALL_GPUS="${ALL_GPUS:-0,1,2,3,4,5,6,7}"

SAVE_FEWSHOT_MODEL="${SAVE_FEWSHOT_MODEL:-0}"
[ "$SAVE_FEWSHOT_MODEL" = "1" ] || SAVE_FEWSHOT_MODEL=""

# ============ Required path configuration (edit for your environment) ============
DINO=/path/to/dinov3-vitl16-pretrain-lvd1689m
TRAIN=/path/to/AIGIBench/train
VAL=/path/to/AIGIBench/train/val
FAKE_BASE=/path/to/Treasure/fake
REAL_VAL=/path/to/Treasure/real


# ============ Output directory (keep the Fleet dir clean; outputs go elsewhere) ============
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/outputs}"
CKPT_DIR="${OUTPUT_ROOT}/checkpoints_q128"
CKPT="${CKPT_DIR}/pretrain_model.pth"
PROTO="${CKPT_DIR}/prototypes"
FS_OUT="${OUTPUT_ROOT}/fewshot_experiment"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "$LOG_DIR" "$CKPT_DIR" "$FS_OUT"

DATASETS=("ADM" "NextStep" "Qwen-Image" "HunyuanImage-3.0" "GPT4O_Image_T2I" "FLUX.2" "StarGAN" "wan2.5-t2i-preview")

# ============ Stage 1: Pretraining (val 98% early stop) ============
echo "############ [$(date +%H:%M:%S)] Stage 1: Pretraining (val 98% early stop) ############"
if [ -f "$CKPT" ] && [ -d "$PROTO" ] && [ "${FORCE_PRETRAIN:-0}" != "1" ]; then
  echo "[Stage1 skip] Pretrained weights already exist: $CKPT (set FORCE_PRETRAIN=1 to retrain)"
else
  CUDA_VISIBLE_DEVICES="$ALL_GPUS" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" -m fleet.train.dual_branch_q128 \
      --dinov3_model_path "$DINO" \
      --aigibench_train "$TRAIN" --aigibench_val "$VAL" \
      --output_dir "$CKPT_DIR" --force_train 1 \
      > "${LOG_DIR}/pretrain.log" 2>&1
  test -f "$CKPT" && test -d "$PROTO" || { echo "[ERROR] Pretrained weights were not generated"; exit 1; }
fi
echo "[Stage1 done $(date +%H:%M:%S)] Pretraining ready"

# ============ Stage 2: Few-shot fine-tuning (d10/lr3e-5) ============
echo "############ [$(date +%H:%M:%S)] Stage 2: Few-shot fine-tuning ############"
run_fs () {
  local dataset="$1"; local ds_safe="${dataset//\./_}"
  local log="${LOG_DIR}/fewshot_${ds_safe}.log"
  echo "------------ [$(date +%H:%M:%S)] dataset=${dataset} ------------"
  CUDA_VISIBLE_DEVICES="$ALL_GPUS" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHON="$PYTHON" PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" \
  FLEET_AIGIBENCH_TRAIN="$TRAIN" FLEET_AIGIBENCH_VAL="$VAL" \
  FLEET_DINOV3_MODEL_PATH="$DINO" \
  FLEET_OTHER_FAKE_BASE_DIR="$FAKE_BASE" FLEET_REAL_VAL_DIR="$REAL_VAL" \
  PRETRAIN_CHECKPOINT="$CKPT" PROTOTYPE_DIR="$PROTO" \
  OUTPUT_BASE_DIR="$FS_OUT" SUBSETS_CSV="$dataset" NUM_EPOCHS=20 BATCH_SIZE=32 \
  DISTILL_WEIGHT=10.0 NUM_WORKERS=16 VAL_BATCH_SIZE=256 \
  SAVE_FEWSHOT_MODEL="$SAVE_FEWSHOT_MODEL" \
  FORCE=1 CONDA_SH= ENV_NAME= \
  bash "${REPO_ROOT}/src/fleet/train/run_fewshot_experiment.sh" > "$log" 2>&1
}
i=0; total=${#DATASETS[@]}
for ds in "${DATASETS[@]}"; do
  i=$((i+1))
  echo ">>>>>>>>>> [${i}/${total}] ${ds} $(date +%H:%M:%S)"
  if run_fs "$ds"; then echo "<<<<<<<<<< [${i}/${total}] done ${ds} $(date +%H:%M:%S)"
  else echo "<<<<<<<<<< [${i}/${total}] FAILED ${ds} $(date +%H:%M:%S) see log"; fi
done
echo "[Stage2 done $(date +%H:%M:%S)] Few-shot done"

# ============ Stage 3: Summary ============
echo "############ [$(date +%H:%M:%S)] Stage 3: Summary ############"
SUMMARY="${LOG_DIR}/run_summary.txt"
FS_TABLE_TMP="${LOG_DIR}/.fs_table.tmp"

# Few-shot table (dynamically scan the datasets actually run; used for both the summary file and the console)
"$PYTHON" -c "
import json,os
base='${FS_OUT}/custom'
# Dynamically scan dataset dirs that actually contain fewshot_results.json
entries=[]
if os.path.isdir(base):
    for ds in os.listdir(base):
        if os.path.isfile(os.path.join(base,ds,'fewshot_results.json')):
            entries.append(ds)
entries.sort()
def fmt(x): return f'{x:6.2f}' if isinstance(x,(int,float)) else '  N/A '
print(f\"{'Dataset':>20} | {'Query':>7} {'Real':>7} {'valF':>7} {'valR':>7}\")
print('-'*66)
qs=rs=vfs=vrs=n=0
for ds in entries:
    p=os.path.join(base,ds,'fewshot_results.json')
    d=json.load(open(p)); fr=d.get('final_results',{}); o={}
    for k,v in fr.items():
        if k.startswith('Query集'): o['q']=v.get('total_acc')
    for key,a in [('final/real','r'),('AIGIBench/val-fake','vf'),('AIGIBench/val-real','vr')]:
        v=fr.get(key)
        if isinstance(v,dict): o[a]=v.get('total_acc')
    q,r,vf,vr=o.get('q'),o.get('r'),o.get('vf'),o.get('vr')
    print(f'{ds:>20} | {fmt(q)} {fmt(r)} {fmt(vf)} {fmt(vr)}')
    if None not in (q,r,vf,vr): qs+=q; rs+=r; vfs+=vf; vrs+=vr; n+=1
if n: print('-'*66); print(f'{\"Mean\":>20} | {fmt(qs/n)} {fmt(rs/n)} {fmt(vfs/n)} {fmt(vrs/n)}')
if not entries: print('(no few-shot results)')
" > "$FS_TABLE_TMP" 2>&1

{
  echo "============================================================"
  echo " Fleet full pipeline summary  $(date)"
  echo "============================================================"
  echo "=== Pretraining validation results (AIGIBench/val) ==="
  "$PYTHON" -c "
import json,os
p='${CKPT_DIR}/validation_results_dual_branch_with_attn_loss_and_coverage_no_residual_q128_freq.json'
if os.path.isfile(p):
    d=json.load(open(p))
    for k,v in d.items():
        print(f'  {k}: total={v.get(\"total_acc\")} fake={v.get(\"fake_acc\")} real={v.get(\"real_acc\")} n={v.get(\"n_samples\")}')
else: print('  (no validation results)')
" 2>&1
  echo; echo "=== Few-shot (Query/Real/valF/valR) ==="
  cat "$FS_TABLE_TMP"
  echo; echo "============================================================"
} > "$SUMMARY" 2>&1

# Also print the few-shot table to the console
echo "=== Few-shot (Query/Real/valF/valR) ==="
cat "$FS_TABLE_TMP"
rm -f "$FS_TABLE_TMP"

echo "DONE" > "${LOG_DIR}/run_done.flag"
echo "[all done $(date +%H:%M:%S)] summary: ${SUMMARY}"
