# Control Model — Run Guide

Every command, in order, from combining the data to the final benchmark table.

**Paths assumed** (match the conventions in `GUIDELINE.md` — adjust if yours differ):

| what | path |
|---|---|
| dataset directory | `data/fineweb_val/` |
| shard checkpoints | `runs/shard_01/final.pt` … `runs/shard_05/final.pt` |
| merged model | `runs/merged.pt` |
| control model (this guide creates it) | `runs/control/` |
| fine-tune results | `runs/finetune/` |
| benchmark report | `runs/benchmark/` |

Each step gives **bash** (Git Bash) first, then **PowerShell**. Both work on your machine.

---

## Step 0 — Preflight

Never train on a machine where the merge math fails.

```bash
python -m pytest tests -q
```

Must print `20 passed`.

```bash
pip install datasets
```

Needed for AG News in Step 3. Everything else is already installed.

---

## Step 1 — Combine the partitions into one dataset

The shards each saw one partition; the control must see all of them.

```bash
python scripts/combine_partitions.py --data-dir data/fineweb_val
```

```powershell
python scripts\combine_partitions.py --data-dir data\fineweb_val
```

Creates `data/fineweb_val/part_all.bin` and `part_all.manifest.json`.
Excludes `val.bin`, `part_XX.val.bin` and `seed.bin` by design — no model may train on
held-out data, and `seed.bin` is a *sample of* the partitions so its documents are
already present.

Note the printed token total; you will sanity-check against it in Step 2.

---

## Step 2 — Train the control model

Same architecture as the merged model, 2 epochs over the whole dataset, checkpoint every
30 minutes.

### 2a. Dry run first — always

```bash
python scripts/train_baseline.py \
  --name control \
  --family configs/family_reference.json \
  --data data/fineweb_val/part_all.bin \
  --val  data/fineweb_val/val.bin \
  --out-dir runs/control \
  --match-merged runs/merged.pt \
  --max-epochs 2 \
  --checkpoint-every-min 30 \
  --batch-size 4 --grad-accum 16 \
  --dry-run
```

```powershell
python scripts\train_baseline.py `
  --name control `
  --family configs\family_reference.json `
  --data data\fineweb_val\part_all.bin `
  --val  data\fineweb_val\val.bin `
  --out-dir runs\control `
  --match-merged runs\merged.pt `
  --max-epochs 2 `
  --checkpoint-every-min 30 `
  --batch-size 4 --grad-accum 16 `
  --dry-run
```

**Before continuing, confirm the printout says:**

```
    -> exact match: the control is an architectural twin.
```

That line means the control's head count, FFN width and parameter count were read from
`runs/merged.pt` and verified against its actual tensors. Also read the
`ESTIMATED optimizer-state memory` line — if it is close to your GPU's capacity, adjust
batch/accum now (see 2c).

### 2b. Train

Same command, `--dry-run` removed.

```bash
python scripts/train_baseline.py \
  --name control \
  --family configs/family_reference.json \
  --data data/fineweb_val/part_all.bin \
  --val  data/fineweb_val/val.bin \
  --out-dir runs/control \
  --match-merged runs/merged.pt \
  --max-epochs 2 \
  --checkpoint-every-min 30 \
  --batch-size 4 --grad-accum 16
```

```powershell
python scripts\train_baseline.py `
  --name control `
  --family configs\family_reference.json `
  --data data\fineweb_val\part_all.bin `
  --val  data\fineweb_val\val.bin `
  --out-dir runs\control `
  --match-merged runs\merged.pt `
  --max-epochs 2 `
  --checkpoint-every-min 30 `
  --batch-size 4 --grad-accum 16
```

Produces `runs/control/final.pt` (portable) and `runs/control/latest.pt` (resumable).

**If the PC crashes, rerun the exact same command.** It resumes from `latest.pt` at the
right step, epoch, token count and data position. Nothing else to do.

### 2c. If you hit CUDA OOM

Halve the batch, double the accumulation. The optimizer batch — and therefore the
training math — is unchanged; only the memory peak drops.

```bash
--batch-size 2 --grad-accum 32
```

### 2d. Token accounting — worth checking once

Two epochs over the union is **exactly token-matched** to the shards if each shard also
ran 2 epochs over its own partition (5 shards × 2 × |D|/5 = 2|D| = 2 epochs × |D|). If
your shards ran different budgets, the comparison is about compute rather than data, and
you should say so. To match the shards' actual tokens instead of using epochs, swap
`--max-epochs 2` for:

```bash
--match-tokens-from runs/shard_01/final.pt runs/shard_02/final.pt runs/shard_03/final.pt runs/shard_04/final.pt runs/shard_05/final.pt
```

Either way, Step 5's accounting table prints both models' token counts so nothing is
hidden.

---

## Step 3 — Fine-tune the merged model

AG News, 3 seeds. Takes roughly an hour per seed on a modern GPU.

```bash
python scripts/finetune.py \
  --init-from runs/merged.pt \
  --name merged \
  --dataset ag_news \
  --out-root runs/finetune \
  --seeds 0 1 2 \
  --max-epochs 3 \
  --batch-size 16 \
  --max-length 256 \
  --checkpoint-every-min 30
```

```powershell
python scripts\finetune.py `
  --init-from runs\merged.pt `
  --name merged `
  --dataset ag_news `
  --out-root runs\finetune `
  --seeds 0 1 2 `
  --max-epochs 3 `
  --batch-size 16 `
  --max-length 256 `
  --checkpoint-every-min 30
```

Writes `runs/finetune/merged_ag_news_seed{0,1,2}/` and
`runs/finetune/merged_ag_news_<hash>_aggregate.json`, and prints mean ± std across seeds.

The first run tokenizes AG News and caches it to `data/finetune_cache/`. Step 4 reuses
that exact cache, so both models see byte-identical inputs.

---

## Step 4 — Fine-tune the control model

**Identical flags**, only `--init-from` and `--name` change. That is what makes the
comparison fair.

```bash
python scripts/finetune.py \
  --init-from runs/control/final.pt \
  --name control \
  --dataset ag_news \
  --out-root runs/finetune \
  --seeds 0 1 2 \
  --max-epochs 3 \
  --batch-size 16 \
  --max-length 256 \
  --checkpoint-every-min 30
```

```powershell
python scripts\finetune.py `
  --init-from runs\control\final.pt `
  --name control `
  --dataset ag_news `
  --out-root runs\finetune `
  --seeds 0 1 2 `
  --max-epochs 3 `
  --batch-size 16 `
  --max-length 256 `
  --checkpoint-every-min 30
```

### 4b. Optional but recommended — one shard, as a lower bound

Shows how much merging actually recovered. A shard saw 1/5 of the data.

```bash
python scripts/finetune.py --init-from runs/shard_01/final.pt --name shard01 --dataset ag_news --out-root runs/finetune --seeds 0 1 2 --max-epochs 3 --batch-size 16 --max-length 256
```

### 4c. Optional but recommended — linear probes

Trains only the classification head, so it measures the frozen representations with no
fine-tuning confound. Fast (minutes, not hours) and far more sensitive to merge damage.

```bash
python scripts/finetune.py --init-from runs/merged.pt --name merged_probe --dataset ag_news --out-root runs/finetune --seeds 0 1 2 --max-epochs 5 --freeze-backbone --head-lr 1e-3 --batch-size 32 --max-length 256
```

```bash
python scripts/finetune.py --init-from runs/control/final.pt --name control_probe --dataset ag_news --out-root runs/finetune --seeds 0 1 2 --max-epochs 5 --freeze-backbone --head-lr 1e-3 --batch-size 32 --max-length 256
```

---

## Step 5 — Benchmark both

```bash
python scripts/benchmark.py \
  --models runs/merged.pt runs/control/final.pt runs/shard_01/final.pt \
  --bins data/fineweb_val/val.bin \
         data/fineweb_val/part_01.val.bin \
         data/fineweb_val/part_02.val.bin \
  --finetune-root runs/finetune \
  --compare merged control \
  --out runs/benchmark \
  --batch-size 8
```

```powershell
python scripts\benchmark.py `
  --models runs\merged.pt runs\control\final.pt runs\shard_01\final.pt `
  --bins data\fineweb_val\val.bin `
        data\fineweb_val\part_01.val.bin `
        data\fineweb_val\part_02.val.bin `
  --finetune-root runs\finetune `
  --compare merged control `
  --out runs\benchmark `
  --batch-size 8
```

Prints three sections and writes `runs/benchmark/benchmark_report.json`:

1. **Language modelling** — perplexity per model per file. The `part_XX.val.bin` columns
   are the specialty-retention check: the merged model should be competitive on *every*
   partition, each shard only on its own.
2. **Downstream** — AG News accuracy and macro-F1, mean ± std over seeds, plus a **95%
   confidence interval on the merged − control difference**. It prints `RESOLVED` if the
   interval excludes zero, `NOT RESOLVED` if it straddles zero. If it says NOT RESOLVED,
   add seeds before claiming a difference.
3. **Accounting** — parameters and pretraining tokens per model, so an unfair comparison
   is visible in the same table as the result.

### 5b. Add the weight-averaging baseline

The strongest contrast in the thesis. Build it once, include it everywhere.

```bash
python scripts/merge.py --inputs runs/shard_0*/final.pt --out runs/avg.pt --method avg --alpha-mode uniform
```

Then add `runs/avg.pt` to the `--models` list in Step 5.

---

## Step 6 — Read the samples by hand

```bash
python scripts/generate.py --model runs/merged.pt --compare runs/control/final.pt --prompt "Photosynthesis is the process by which" --seed 0 --max-new-tokens 120
```

```powershell
python scripts\generate.py --model runs\merged.pt --compare runs\control\final.pt --prompt "Photosynthesis is the process by which" --seed 0 --max-new-tokens 120
```

Same prompt, same seed, both models side by side. Interactive session: drop `--prompt`.

---

## Full sequence, copy-paste

```bash
python -m pytest tests -q
pip install datasets
python scripts/combine_partitions.py --data-dir data/fineweb_val
python scripts/train_baseline.py --name control --family configs/family_reference.json --data data/fineweb_val/part_all.bin --val data/fineweb_val/val.bin --out-dir runs/control --match-merged runs/merged.pt --max-epochs 2 --checkpoint-every-min 30 --batch-size 4 --grad-accum 16 --dry-run
python scripts/train_baseline.py --name control --family configs/family_reference.json --data data/fineweb_val/part_all.bin --val data/fineweb_val/val.bin --out-dir runs/control --match-merged runs/merged.pt --max-epochs 2 --checkpoint-every-min 30 --batch-size 4 --grad-accum 16
python scripts/finetune.py --init-from runs/merged.pt --name merged --dataset ag_news --out-root runs/finetune --seeds 0 1 2 --max-epochs 3 --batch-size 16 --max-length 256 --checkpoint-every-min 30
python scripts/finetune.py --init-from runs/control/final.pt --name control --dataset ag_news --out-root runs/finetune --seeds 0 1 2 --max-epochs 3 --batch-size 16 --max-length 256 --checkpoint-every-min 30
python scripts/benchmark.py --models runs/merged.pt runs/control/final.pt --bins data/fineweb_val/val.bin --finetune-root runs/finetune --compare merged control --out runs/benchmark --batch-size 8
```

---

## Checklist before reporting numbers

- [ ] `pytest tests -q` printed `20 passed` on this machine
- [ ] Step 2a printed `-> exact match: the control is an architectural twin.`
- [ ] Control trained on `part_all.bin`, never on `val.bin`
- [ ] Merged and control fine-tuned with **identical** flags except `--init-from`/`--name`
- [ ] At least 3 seeds per model
- [ ] Step 5's accounting table shows comparable parameters and token counts
- [ ] Weight-averaging baseline included in the table

---

## What to expect

The control will probably win on both perplexity and AG News accuracy. Plan for that —
SAP-Pure is structurally an ensemble folded into one network, bounded above by the
ensemble of its shards, with the composition gap on top.

The results that hold regardless:

* merged ≫ any individual shard — merging recovered real capability
* merged ≫ weight-averaging — the central hypothesis of report §12
* merged vs control gap — the quantity a healing pass (§6.5b) is supposed to close, and
  reporting SAP-Pure and SAP-Heal as separate rows makes the gap measurable

---

See also: [BASELINE_GUIDE.md](BASELINE_GUIDE.md) (budgets, memory, crash recovery in
depth) and [FINETUNE_GUIDE.md](FINETUNE_GUIDE.md) (changing datasets, hyperparameters,
benchmark selection).
