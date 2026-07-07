# SAP — The Complete Usage Guideline

One document, start to finish: prepare a dataset once → train shards anywhere (one PC or
many) → merge → evaluate. Includes the multi-PC transport checklist, both validation
modes (global + per-shard), memory tuning for any GPU, and troubleshooting for every
failure we have actually hit.

---

## 0. The mental model (30 seconds)

| term | meaning |
|---|---|
| **family** | the fixed skeleton every model shares: `n_layers, d_model, d_head, vocab_size, max_seq_len, rope_theta, norm_eps` — one JSON file, decided once |
| **partition** | one slice of the dataset (`part_01.bin` …), made once at prep time |
| **shard** | one small model trained on one partition, fully independently |
| **merge** | tensor surgery that stacks all shards' heads + neurons into one wider model; gated by an exactness check |
| **val sets** | `val.bin` = global held-out (all models compared on it); `part_XX.val.bin` = per-partition held-out (specialty checks) |

Key fact for planning: **the merged big model is never trained** — it is assembled on
CPU and only evaluated (forward passes). So the *training* memory budget is always the
*shard* size, not the final model size. Training 171M shards on a 24 GB card today is
the same job in the real 1B run; only the number of shards and hours change.

## 1. One-time setup (per machine)

```bash
cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build
pip install -r requirements.txt
python -m pytest tests -q     # MUST print "20 passed" on EVERY machine you use
```

The tests verify the merge math on that machine. Never train on a box where they fail.

## 2. Prepare the dataset — ONCE, on one machine (CPU-only step)

Tokenization is inherently a CPU job (text → integer IDs; no GPU math involved). Do it
once; the outputs are plain binary files that work on every machine forever.

```bash
python scripts/prepare_data.py \
  --inputs hf:HuggingFaceFW/fineweb-edu:sample-10BT:train \
  --tokenizer gpt2 --tokenizer-backend tiktoken \
  --out-dir data/fineweb_val --num-partitions 5 \
  --val-fraction 0.005 --partition-val-fraction 0.001 \
  --max-tokens 4000000000
```

Notes:
- `--tokenizer-backend tiktoken` — use this. Identical GPT-2 token IDs to HuggingFace,
  but stable (the HF Rust tokenizer segfaults on long runs) and ~5–30× faster.
- Inputs can be `hf:` streams, local `.jsonl`/`.txt`, or (if present in your version)
  local parquet. `--partition-mode by-input` = domain partitioning (file k → partition k).
- On a shared PC, be polite: `nice -n 19 python scripts/prepare_data.py ...`

**What you get in `data/<name>/`:**

| file | role | trained on? |
|---|---|---|
| `part_01.bin` … `part_NN.bin` | training partitions D₁…D_N | yes (shard k on part k) |
| `val.bin` | **global** held-out set — the common yardstick for ALL models | never |
| `part_XX.val.bin` | per-partition held-out — specialty retention checks | never |
| `seed.bin` | mixed sample for the optional seed phase (only if `--seed-fraction > 0`) | seed only |
| `meta.json` | tokenizer name, vocab, dtype, token counts — **always travels with the bins** | — |

Sanity check after prep: the printed `vocab_size` must equal `vocab_size` in your family
JSON (gpt2 → 50257). The trainer enforces this, but check early.

## 3. Moving work to other PCs (asynchronous training)

**Q: Do I prepare the dataset once and carry a partition to another PC?**
Yes — exactly. Tokenize once, then each training PC gets only its own slice. Never
re-tokenize; a `.bin` is byte-identical everywhere.

**Copy-to-another-PC checklist (for shard k):**

1. the repository (same version — run `pytest` there: 20 passed);
2. `configs/<family>.json` — must be byte-identical on every machine;
3. `data/<name>/part_k.bin` **and** `data/<name>/meta.json` (meta must sit next to the
   bin — it carries the dtype and vocab). Optionally `val.bin` for during-training evals;
4. `runs/seed/final.pt` — only if this shard branches from a seed.

**Q: What about the tokenizer — how do I take it?**
You don't. Training never touches the tokenizer — the `.bin` files are already token
IDs, and `meta.json` records which tokenizer made them. The tokenizer is needed only
(a) at prep time and (b) if you want to *generate text* during evaluation
(`scripts/evaluate.py --generate --tokenizer gpt2`, fetched by name automatically).

When shard k finishes, carry back **one file**: `runs/shard_k/final.pt` (optimizer state
stripped — it's the small, portable one). Merge on any machine holding all the finals.

## 4. Training shards

Single-PC-overnight (the validation runner does all of this automatically — §6). For
manual / multi-PC training, one command per shard:

```bash
python scripts/train.py --name shard_03 --family configs/family_reference.json \
  --data data/fineweb_val/part_03.bin --val data/fineweb_val/val.bin \
  --out-dir runs/shard_03 --n-heads 4 --d-ff 512 --init-seed 3003 \
  --batch-size 8 --grad-accum 16 --seq-len 1024 \
  --max-hours 12 --checkpoint-every-min 60
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

## 5. Merging and evaluating

```bash
# the method (token-weighted alphas, exact function average):
python scripts/merge.py --inputs runs/shard_0*/final.pt --out runs/merged.pt
# the no-scaling condition (exact function sum):
python scripts/merge.py --inputs runs/shard_0*/final.pt --out runs/merged_uns.pt --no-scale
# the naive baseline:
python scripts/merge.py --inputs runs/shard_0*/final.pt --out runs/avg.pt --method avg --alpha-mode uniform
# continual merge (a merged model is a normal family member; α bookkeeping automatic):
python scripts/merge.py --inputs runs/merged.pt runs/shard_06/final.pt --out runs/merged2.pt
```

Every stack merge must pass its built-in exactness gate (block-by-block, float64) or it
refuses to save. Expect reported errors ~1e-8 for fp32 weights.

**Evaluation — you have BOTH validation modes, choose per command:**

```bash
# (a) EVERYTHING on the SAME overall set (big and small models, apples-to-apples):
python scripts/evaluate.py \
  --models runs/shard_0*/final.pt runs/merged.pt runs/avg.pt \
  --data data/fineweb_val/val.bin --out results/global.json

# (b) each small model on ITS OWN partition's held-out set:
python scripts/evaluate.py --models runs/shard_01/final.pt --data data/fineweb_val/part_01.val.bin
python scripts/evaluate.py --models runs/shard_02/final.pt --data data/fineweb_val/part_02.val.bin

# (c) the full grid — every model x every set at once (what validate.py does):
python scripts/evaluate.py \
  --models runs/shard_0*/final.pt runs/merged.pt \
  --data data/fineweb_val/val.bin data/fineweb_val/part_0*.val.bin --out results/grid.json
```

Headline thesis numbers come from (a); (b)/(c) answer "did the merge keep each shard's
specialty". All these sets are held out from all training, so every comparison is honest.

## 6. The hands-off validation runner (one PC, fixed time window)

```bash
tmux new -s sap
python validation/validate.py        # then Ctrl-b d, walk away
```

Prep (skipped if done) → trains all shards back-to-back, dividing the remaining budget
evenly → 3 merges → full eval grid. Reboot-safe: rerun the same command; finished shards
skip, the interrupted one resumes. Results in `validation/results/`
(`eval_table.txt`, `run_summary.json` with per-shard epochs/steps/tokens and per-merge
exactness errors, `progress.log`). All knobs are in the CONFIG block at the top of
`validate.py`.

## 7. Memory: what to change on ANY GPU

The trainer's peak GPU memory is governed by a few knobs — none are 4090-specific:

```
tokens per optimizer step = batch_size × grad_accum × seq_len   (the "math")
peak GPU memory          ≈ f(batch_size, seq_len, model width)  (the "hardware")
```

**Golden rule: tune `batch_size` to the card, then set `grad_accum` so
batch_size × grad_accum stays constant.** Identical training math, different memory.

| knob | to use LESS memory | to use MORE (bigger/faster) | effect |
|---|---|---|---|
| `batch_size` | halve it | raise until near the VRAM limit | linear in peak memory — the main dial |
| `grad_accum` | double it (compensates batch) | lower it | no memory effect; keeps math identical |
| `seq_len` | lower (e.g. 512) | raise up to family `max_seq_len` | ~linear in memory |
| shard width (`--n-heads/--d-ff`) | thinner shards | wider shards | model+optimizer+activations |
| `compile` | keep off | on (≈20–40% faster) once the driver is proven stable | compile-time memory spike |
| `eval_batches` / `eval_interval` | fewer/less often | — | eval spikes memory briefly |

Starting points at seq_len 1024, ~171M shard, bf16 (measure, then adjust):

| card | batch_size × grad_accum | expected peak |
|---|---|---|
| 24 GB (4090 / A5000) | **8 × 16** (default) | ~10 GB — safe headroom |
| 32 GB (5090) | 16 × 8 | ~19 GB |
| 48 GB (A6000) | 24–32 × 4–5 | ~28–38 GB |

Why we leave ~50% headroom rather than filling the card: with a 50k vocab, each forward
materializes a `(batch, seq, vocab)` logits tensor (~0.8 GB at batch 8, ~1.6 GB at batch
16) plus fp32 copies inside the loss — and allocator *fragmentation* (the
"reserved but unallocated" in OOM messages) means you effectively lose 10–25% of VRAM
over a long run. Running at 90% of the card is how you OOM at hour 3. Also already
built in: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set automatically by
`validate.py` and `scripts/train.py`), logits freed immediately after the loss, cache
release after every in-training eval, and full GPU cleanup between shards.

To watch live: `watch -n 2 nvidia-smi` during the first 10 minutes of a run; peak
usually appears within the first eval + checkpoint cycle.

## 8. Troubleshooting (everything we have actually hit)

| symptom | cause | fix |
|---|---|---|
| `CUDA out of memory` mid-training, big "reserved but unallocated" | fragmentation near the ceiling (often right after an eval) | lower `batch_size` (+raise `grad_accum`); the allocator env var + cache releases are already in the code |
| every later shard OOMs instantly after one fails | (fixed) a failed shard's exception kept its model alive on GPU | fixed in `validate.py` (`gc.collect` + `empty_cache` between shards) — update your copy |
| segfault during tokenization | HF Rust tokenizer instability | use `--tokenizer-backend tiktoken` |
| crash *after* "dataset prepared" table prints | harmless HF-streaming teardown bug; data is complete | ignore, or avoid by preparing from local files |
| `Fatal Python error: PyGILState_Release` at exit | same as above | same |
| `resume mismatch on 'batch_size' ...` | you changed a resume-critical setting mid-run | intended guard; `rm -rf` that run dir (or `--fresh`) to restart it under the new settings |
| whole SYSTEM freezes/reboots | driver/hardware — user-space Python cannot crash Linux | reboot cleanly after any driver repair; check `dmesg -T | tail -50` after reboot (OOM-killer? Xid errors?); watch temps under load |
| `Driver/library version mismatch` from nvidia-smi | driver updated while old kernel module loaded | reboot |
| torch says `cuda available: False` | driver problem, not code | fix driver first; never start a run without `torch.cuda.is_available()` → True |
| segfault on the first training step with `compile: true` | Triton kernel compilation on a fragile driver | keep `compile: false` until the box has survived a full run |

## 9. Scaling to the real (1B) run

Same pipeline, bigger numbers — nothing new to learn:

1. Prep once with `--num-partitions 10` and 30–40B tokens (see `data/PREPARE_FINEWEB.md`
   if present, or just raise `--max-tokens`).
2. Train 10 shards (same `--n-heads 4 --d-ff 512` ≈ 171M each) for ~12h each — on one PC
   sequentially across nights, or on several PCs at once (§3).
3. One merge command → ~1B model. One eval command → the thesis table.

The merged 1B model needs only ~2–4 GB for forward-pass evaluation — evaluation of the
big model is *lighter* than training a shard. The hard part of the whole project is the
per-shard training you are already validating.
