# Validation run — one Python file, no seed, run and walk away (Linux)

Goal: prove the whole SAP pipeline works end-to-end on real data, unattended, inside a
single 12-hour window on one 4090 — before the full 40B / 1B run. **No seed this run**
(every shard trains independently from scratch); a seed-based run can be a separate day.

There are **no shell scripts**. You run one Python file and walk away:

```bash
python validation/validate.py
```

That single command does all four phases by itself:

```
prepare ~4B tokens of FineWeb-Edu -> 5 partitions (+ held-out val)   [skipped if already done]
   |
train shard_01 -> shard_02 -> ... -> shard_05   (each from scratch, back-to-back,
   |             the next starts the instant the previous finishes; remaining time
   |             is divided evenly so it all fits the 12h budget)
   v
merge:  merged_scaled   (token-weighted, exact average — the SAP method)
        merged_unscaled (exact sum — the no-scaling condition)
        baseline_avg    (naive weight averaging — the baseline that should lose)
   v
evaluate every model on held-out val -> loss / perplexity / accuracy grid
```

**Model sizes** (reference family L=24, d_model=1536, vocab=50257): each shard is
`H=4, d_ff=512` ≈ **171M**; the 5-way merge ≈ **550M**. Same family and shard config as
the real 1B run — this is a faithful dress rehearsal, just with less training time.

---

## Step 0 — setup (once)

```bash
cd /path/to/SAS
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build
pip install -r requirements.txt
python -m pytest tests -q          # MUST print "20 passed" — verifies the merge math here
```

Disk: ~8GB for the ~4B token files + ~15GB for checkpoints. Have ~30GB free.

## Step 1 — run it (inside tmux, then walk away; ~12h)

```bash
tmux new -s sap
python validation/validate.py
#   detach:  Ctrl-b  then  d      reattach later:  tmux attach -t sap
```

The first ~15–25 min tokenizes ~4B tokens of FineWeb-Edu (this step needs network), then
it trains the 5 shards, merges, and evaluates. All of it counts against the 12h budget,
and the runner guarantees it never overruns your access window.

**If the machine reboots or you stop it:** run the exact same command again. Finished
shards are skipped, an interrupted shard resumes from its last checkpoint (saved every
20 min), and it continues. Nothing is retrained.

> Prefer to tokenize during the day and keep all 12h for training? Run the same command
> once; after the log says `prep done`, press Ctrl-C. Tonight, run it again — it sees the
> prepared data, skips prep, and spends the whole budget training.

## Step 2 — read the results (morning)

Everything is in `validation/results/`:

| file | contents |
|---|---|
| `eval_table.txt` | the headline grid: every model's loss / perplexity / accuracy on `val.bin` and each partition's held-out set |
| `run_summary.json` | per-shard **epochs, steps, tokens, elapsed, stop reason**; per-merge **exactness error + alphas**; the eval table |
| `progress.log` | timestamped timeline of the whole night |
| `runs/<shard>/train_log.jsonl` | per-step loss/lr/val curves (plot these) |

### What "success" looks like

It's a **validation**, so the bar is "the pipeline works and the math holds," not great
perplexity (each shard trains ~2h, far below optimal — deliberately undertrained). Check:

1. **All 5 shards finished** — `run_summary.json` shows `status: done`, `epochs_completed ≥ 1`,
   and a falling loss in each `train_log.jsonl`. Proves training + auto-chaining +
   checkpointing.
2. **Both stack merges passed their exactness gate** — `merged_scaled` and
   `merged_unscaled` show `verify_max_err` around 1e-8 or smaller. **This is the core
   thesis claim holding on real trained weights.**
3. **The merged model is coherent** — `merged_scaled` perplexity is in the same ballpark
   as the shards (each saw only 1/5 of the data, so landing *near* them is the expected
   SAP-Pure result).
4. **The stack merge beats the naive baseline** — `merged_scaled` perplexity should be
   below `baseline_avg`. First data point for the "graceful vs. catastrophic" hypothesis.

If all four hold, scale up to the real run (`data/PREPARE_FINEWEB.md`).

---

## Changing anything — edit the `CONFIG` block at the top of `validate.py`

No separate config files. The knobs live in a clearly-marked `CONFIG` dict at the top of
[validate.py](validate.py):

- **Out of GPU memory?** Lower `batch_size` (try 12, then 8), raise `grad_accum` to keep
  `batch_size × grad_accum × seq_len` ≈ 130k tokens/step.
- **More or fewer shards?** Change `num_shards` (the runner partitions the data to match).
- **More/less data prepared?** Change `prepare_max_tokens` (default 4B is sized so each
  ~2h shard trains on <1 epoch, i.e. no heavy repetition).
- **Different budget?** Change `total_hours`.
- **Want the seed-based run tomorrow?** That's the other path (branch each shard from a
  short seed). Use `scripts/train.py --init-from runs/seed/final.pt ...` per the top-level
  README, or ask and I'll add a seed-mode flag to this runner.

## How the time-budgeting works (so you can trust it)

The runner marks a start time (covering prep + training). Before each shard it computes
`remaining = total_hours − reserve − elapsed` and gives the shard
`remaining / (shards not yet trained)` — so if prep ran long, or a shard was interrupted,
the rest automatically re-balance to still fit. If time runs out, remaining shards are
skipped and it goes straight to merging + evaluating whatever finished. It **always**
produces a merge and an eval and **never** exceeds `total_hours`.

The merge math is untouched: the runner calls the same gated `merge_checkpoints` as the
CLI, so an incorrect merge would fail its exactness gate and not be saved.
