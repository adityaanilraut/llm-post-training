# Qwen 0.5B Tool-Calling SFT (MLX, Apple Silicon)

Teach `Qwen2.5-0.5B-Instruct` to emit strict `<tool_call>` JSON for two tools
(`get_weather`, `calculator`), on an 8GB Mac. Pure `.py`, one chat template everywhere.

**Fused model:** https://huggingface.co/rautaditya/qwen2.5-0.5B-toolcall-mlx

## Run

```bash
pip install -r requirements.txt
python prepare_data.py   # builds data/ (101 train / 11 valid / 4 test)
./train.sh               # LoRA rank 16, 800 iters
python eval.py --adapter ./adapters
python eval_benchmark.py --adapter ./adapters
python agent.py --adapter ./adapters --prompt "Compute 128 * 6"
```

## Results

8-case eval — base `0.688` → LoRA **`1.000`**.
22-case held-out benchmark — base `0.682` → LoRA **`0.864`**
(sweeps calc/clarify/no-tool; weak on unseen city spellings, see `eval_benchmark_results.json`).

Shipped weights = iter-300 (best val loss 0.102), fused and on the Hub.
`adapters/` and `fused/` are gitignored — too big for git.

## Files

`prepare_data.py` builds data · `train.sh` trains · `eval.py` / `eval_benchmark.py` benchmark ·
`agent.py` mock-tool agent · `data/` jsonl + result jsons included for reference.
