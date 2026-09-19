#!/bin/bash
# MLX LoRA fine-tune for 8GB Apple Silicon. Stops fast if OOM: lower --batch-size to 1.
set -e
cd "$(dirname "$0")"

# Full-precision base fits on 8GB at 0.5B scale (~3-4GB peak with batch 1).
# 4-bit fallback (more RAM headroom, but measurably worse JSON validity):
#   MODEL=mlx-community/Qwen2.5-0.5B-Instruct-4bit ./train.sh
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

mlx_lm.lora \
  --model "$MODEL" \
  --train \
  --data ./data \
  --adapter-path ./adapters \
  -c ./adapter_config.yaml \
  --batch-size 1 \
  --num-layers 16 \
  --learning-rate 1e-4 \
  --iters 800 \
  --steps-per-report 20 \
  --steps-per-eval 50 \
  --save-every 200 \
  --max-seq-length 1024
# NOTE: no --mask-prompt here. Our data is pre-rendered {"text": ...} with the
# official <tools> template baked in, and masking only works on chat datasets.
# Training on full text is slightly wasteful but harmless at this scale.

echo "Done. Test with: python3 eval.py --adapter ./adapters"
echo "Fuse with: mlx_lm.fuse --model $MODEL --adapter-path ./adapters --save-path ./fused"
