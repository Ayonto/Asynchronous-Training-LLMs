


## Prepare Data

windows 

```powershell
python scripts\prepare_data.py ^
  --inputs hf:HuggingFaceFW/fineweb-edu:sample-10BT:train ^
  --tokenizer gpt2 --tokenizer-backend tiktoken ^
  --out-dir data\fineweb_val --num-partitions 5 ^
  --val-fraction 0.005 --seed-fraction 0.05 ^
  --partition-val-fraction 0.001 ^
  --max-tokens 4000000000
```



## Train Seed


Train seed for 1 epoch on seed partition


```powershell

python scripts\train.py --name seed --family configs\family_reference.json ^
  --data data\fineweb_val\seed.bin --val data\fineweb_val\val.bin ^
  --out-dir runs\seed --n-heads 4 --d-ff 512 --init-seed 3003 ^
  --batch-size 8 --grad-accum 16 --seq-len 1024 ^
  --max-epochs 1 --checkpoint-every-min 30

```

## Training Shard 1

```powershell

python scripts\train.py --name shard_01 --family configs\family_reference.json ^
  --data data\fineweb_val\part_01.bin --val data\fineweb_val\val.bin ^
  --out-dir runs\shard_01 --init-from runs\seed\final.pt ^
  --batch-size 8 --grad-accum 16 --seq-len 1024 ^
  --max-hours 12 --checkpoint-every-min 30

```



- **Budgets:** `--max-epochs` (your preference) / `--max-hours` (cumulative across
  crashes) / `--max-steps` — first one hit wins.
- **Checkpoint cadence is per shard:** `--checkpoint-every-min 60` for one, `180` for
  another, or `--checkpoint-every-steps N`.
- **Crash?** Rerun the *identical* command — it resumes exactly (same batches, same
  optimizer state; test-verified). Changing data/batch/seq-len mid-run is refused; use
  `--fresh` to restart deliberately.
- **With seed:** add `--init-from runs/seed/final.pt` and drop `--n-heads/--d-ff`
  (inherited), or give larger widths for function-preserving growth.
- **Without seed:** give every shard a *different* `--init-seed`.
- LR schedule: `cosine` needs a known horizon (`--max-steps` or `--max-epochs`); with
  only `--max-hours` it falls back to constant automatically (the warning is expected).

