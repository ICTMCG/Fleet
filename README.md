# Fleet: Few Shots Lead Effective AI-generated Image Detection
## Accepted by ICML 26!
# Paper on Arxiv
https://arxiv.org/abs/2606.31082
# Treasure dataset
Apply for access on huggingface: https://huggingface.co/datasets/ThreeLiu/Treasure
# Fleet code
## Run

1. Open `Run.sh` and fill in the paths under the "Required path configuration" section at the top of the file (DINOv3 weights, AIGIBench train/val sets, few-shot fake/real data directories, etc.).
2. Make sure `PYTHON` points to your Python interpreter and `ALL_GPUS` lists the GPUs to use.
3. Run:

```bash
bash Run.sh
```

`Run.sh` runs three stages in sequence: Stage 1 pretraining -> Stage 2 few-shot fine-tuning -> Stage 3 summary. Logs, checkpoints, and the summary are written under `OUTPUT_ROOT` (configured in `Run.sh`).