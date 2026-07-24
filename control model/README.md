# SAP — Stack-and-Scale Asynchronous Pretraining

Pipeline for the thesis *"Asynchronous Training of Decoder-Only Language Models Through
Data Partitioning and Model Merging"* (BRAC University, CSE400).

The workflow: partition a dataset D into D₁…D_N → train one small model per partition,
completely independently (any machine, any time, zero communication) → structurally merge
them into one wider model whose every sublayer computes the exact weighted combination of
the parents' sublayer functions.

**This is not a demo.** Every merge is gated by a built-in exactness check (block-by-block,
float64) that refuses to save a model that doesn't satisfy the math, and the test suite
proves the core identities to ~1e-13 (`python -m pytest tests -q`, 20 tests, seconds, CPU).

---

## Repository layout

```
sap/                    the library
  config.py             family skeleton (fixed) + per-model widths (free)
  model.py              the transformer (RMSNorm, RoPE, SwiGLU, tied embeddings) + checkpoint I/O
  data.py               tokenize/partition any corpus into .bin files; deterministic resumable sampler
  train.py              crash-safe trainer: per-model checkpoint cadence, epoch/time/step budgets
  merge.py              the Stack-and-Scale operator, scaled & unscaled, + exactness gate + avg baseline
  evaluate.py           loss/perplexity/accuracy on any .bin; results grid; text generation
scripts/                the CLIs you actually run
  prepare_data.py       corpus -> val.bin, seed.bin, part_01.bin ... part_NN.bin, meta.json
  train.py              train one model (seed or shard) on one .bin
  merge.py              merge N checkpoints (stack scaled / stack unscaled / naive-average baseline)
  evaluate.py           any set of models x any set of .bin files -> results table + JSON
tests/test_exactness.py the mathematical ground truth as executable tests
configs/                example family JSONs
```

## Install

On the university Linux machine (RTX 4090):

```bash
git clone <this repo>   # or copy the folder
cd SAS
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build
pip install -r requirements.txt
python -m pytest tests -q        # 20 passed = the math is intact on this machine
```

Run the test suite on **every machine** you train on, after every code change. If a test
fails, do not train — the thesis claims rest on those identities.

---

## The five steps

### Step 0 — Decide the family (once, before anything else)

Every model that will ever be merged must share one *family skeleton*, stored in a JSON
file (see `configs/`):

| field | meaning | fixed because |
|---|---|---|
| `n_layers` | transformer blocks | block i of A merges with block i of B |
| `d_model` | residual stream width | all heads/neurons read and write this bus |
| `d_head` | per-head dimension | heads must be uniform to stack |
| `vocab_size` | tokenizer vocabulary | embeddings are averaged; ids must mean the same string |
| `max_seq_len` | context length | RoPE cache size, training window |
| `rope_theta`, `norm_eps` | numerics | must be identical for exactness |

Free per model (what merging adds up): head count `--n-heads` and FFN width `--d-ff`,
plus everything about training (data partition, duration, hardware, batch size, schedule).

**Tokenizer flexibility:** yes, fully flexible — you can use any HuggingFace tokenizer
(`gpt2`, `EleutherAI/gpt-neox-20b`, a Llama tokenizer, or one you train yourself with
`tokenizers`/`sentencepiece` saved in HF format). But it is part of the skeleton: pick it
once, set `vocab_size` in the family JSON to the tokenizer's size (printed by
`prepare_data.py`), and never change it. "Embedding" is not something you pick separately —
the embedding table is a trained part of each model (`vocab_size x d_model`) and the
pipeline handles it; only its *shape* comes from your tokenizer + `d_model` choices.

Sizing intuition for one RTX 4090 (24 GB): shards around 100–200M parameters train
comfortably in bf16. With the reference family (`configs/family_reference.json`,
L=24, d_model=1536, d_head=64): a 4-head/1024-d_ff shard ≈ 150M stackable + embedding.

### Step 1 — Prepare the data (any dataset)

You only need this once, on one machine. Everything downstream reads the produced `.bin`s.

```bash
# random i.i.d. partitions (the standard condition):
python scripts/prepare_data.py \
    --inputs hf:HuggingFaceFW/fineweb-edu:sample-10BT:train \
    --tokenizer gpt2 \
    --out-dir data/fineweb \
    --num-partitions 5 \
    --val-fraction 0.005 \
    --seed-fraction 0.05 \
    --partition-val-fraction 0.002
```

Input formats: `.jsonl` (text under `--text-key`), `.txt` (one document per line, or
`--txt-mode file`), or `hf:name[:config][:split]` for any HuggingFace dataset (streamed,
so it works for corpora larger than RAM; cap with `--max-docs`).

Partition modes:
- `random` (default) — each document is routed to a partition by a seeded hash: i.i.d.
  partitions, the "easy mode" condition.
- `by-input` — input file k becomes partition k: true domain partitioning ("hard mode",
  e.g. `--inputs code.jsonl prose.jsonl math.jsonl --partition-mode by-input`).
- `blocks` — contiguous blocks of documents round-robined: in between.

Output of the example above, in `data/fineweb/`:

| file | role |
|---|---|
| `val.bin` | **global held-out validation** — carved out *before* partitioning; no model ever trains on it; this is what all headline perplexities are measured on |
| `part_01.bin` … `part_05.bin` | the training partitions D₁…D₅ |
| `part_01.val.bin` … | per-partition held-out — for "did the merged model keep shard k's specialty?" |
| `seed.bin` | mixed sample from all partitions (5% here) for the optional seed phase |
| `meta.json` | tokenizer name, vocab size, dtype, token counts — travels with the bins |

**Should you create validation data? Yes — and this pipeline forces it.** The rule:
validation text must never appear in any training file, including `seed.bin`. That is
guaranteed here because the val split is removed first, at the document level, before
partitions and the seed sample are drawn. Use `val.bin` for every comparison between
shards / merged / baselines; use `part_XX.val.bin` for specialty-retention analysis.
Roughly 0.5% of documents (or ~5–20M tokens) is plenty for stable perplexity.

The routing is a pure function of `(document index, --routing-seed)`, so re-running the
same command reproduces identical partitions on any machine.

### Step 2 (optional) — Train the seed

Both modes of your experiment are supported: **with seed** (branch every shard from a
short warm-up) and **without seed** (every shard fully independent from scratch).

```bash
python scripts/train.py --name seed --family configs/family_reference.json \
    --data data/fineweb/seed.bin --val data/fineweb/val.bin \
    --out-dir runs/seed --n-heads 4 --d-ff 1024 \
    --max-epochs 1 --checkpoint-every-min 30
```

The seed budget is the branch-point ablation axis (0% / 5% / 10% of total tokens —
control it via `--seed-fraction` at prep time and the seed's `--max-*` budget).

### Step 3 — Train the shards (asynchronously, anywhere)

Each shard is one command, on any machine, at any time. Examples of every option you asked for:

```bash
# WITH seed, same widths as seed (widths inherited automatically),
# checkpoint every 1 hour, budget = 2 full epochs over its partition:
python scripts/train.py --name shard_01 --family configs/family_reference.json \
    --data data/fineweb/part_01.bin --val data/fineweb/val.bin \
    --out-dir runs/shard_01 --init-from runs/seed/final.pt \
    --checkpoint-every-min 60 --max-epochs 2

# WITH seed but WIDER than the seed (function-preserving growth:
# new heads/neurons start silent and differentiate during training):
python scripts/train.py --name shard_02 --family configs/family_reference.json \
    --data data/fineweb/part_02.bin --out-dir runs/shard_02 \
    --init-from runs/seed/final.pt --n-heads 8 --d-ff 2048 \
    --checkpoint-every-min 180 --max-epochs 2

# WITHOUT seed (independent from scratch — give each shard its own init seed!),
# budget = wall-clock time instead of epochs:
python scripts/train.py --name shard_03 --family configs/family_reference.json \
    --data data/fineweb/part_03.bin --out-dir runs/shard_03 \
    --n-heads 4 --d-ff 1024 --init-seed 3003 \
    --checkpoint-every-min 60 --max-hours 12
```

Budgets — set any of `--max-epochs` (full passes over the partition, your likely choice),
`--max-hours` (cumulative wall time, survives restarts), `--max-steps` (optimizer steps);
whichever is hit first stops the run and writes `final.pt`.

**Crash recovery (the whole point of `latest.pt`):** if the PC dies, just rerun the *same
command*. The trainer finds `out_dir/latest.pt` and resumes with identical optimizer
state, RNG, token counts, cumulative wall time, and — because data order is a pure
function of `(data file, seq len, batch size, data seed, epoch, cursor)` — the exact
batches the uninterrupted run would have seen. This is covered by a test
(`test_crash_resume_reproduces_uninterrupted_run`). Checkpoint cadence is per shard:
`--checkpoint-every-min 60` for m1, `180` for m2, or `--checkpoint-every-steps N` if you
prefer step-based cadence. `--fresh` deliberately restarts; changing data/batch/seq-len
mid-run is refused with an explanation.

One caveat: with the default cosine schedule the learning rate is a function of the
*configured* horizon, so extending `--max-epochs`/`--max-steps` after the fact changes
future LRs (resume itself stays exact). If you expect to extend budgets, use
`--schedule constant`.

Practical 4090 settings: bf16 is on automatically; start around `--batch-size 16
--grad-accum 4 --seq-len 1024` for a ~200M shard and adjust to memory; add `--compile`
for a significant speedup once things are stable.

### Step 4 — Merge

Bring the shards' `final.pt` files to one machine:

```bash
# the thesis default: token-weighted alphas, scaled (exact function AVERAGE):
python scripts/merge.py --inputs runs/shard_01/final.pt runs/shard_02/final.pt \
    runs/shard_03/final.pt --out runs/merged_123.pt

# your "without scaling" condition (exact function SUM — no alpha on wo/w_down):
python scripts/merge.py --inputs ... --out runs/merged_unscaled.pt --no-scale

# alpha control: --alpha-mode tokens (default) | uniform | manual --alphas 0.5 0.3 0.2

# continual merging: a merged model is a normal family member; token bookkeeping
# (alpha = accumulated tokens) is automatic from checkpoint metadata:
python scripts/merge.py --inputs runs/merged_123.pt runs/shard_04/final.pt \
    --out runs/merged_1234.pt

# the naive weight-averaging baseline for the graceful-vs-catastrophic comparison:
python scripts/merge.py --inputs A.pt B.pt --out avg.pt --method avg --alpha-mode uniform
```

Every stack merge runs the exactness gate before saving: merged sublayer vs. the weighted
combination of the original parents, block by block in float64. Typical reported error is
~1e-8 for float32 weights (pure float32 storage rounding; the float64 unit tests pin the
identity itself at ~1e-13). A wrong axis/scale/slice produces errors of order 1 and the
merge refuses to save. Note the honest limitation (thesis §composition gap): exactness is
per sublayer; the whole-model output is only approximately the parents' average, which is
precisely the thing your experiments measure.

Alphas in `--no-scale` mode: the wo/w_down scaling is skipped, but embeddings can't be
summed (they'd double the input magnitude), so they are still combined with the convex
alphas. This is stated in the merged checkpoint's metadata.

### Step 5 — Evaluate (the dynamic test harness)

Any set of models × any set of token files:

```bash
python scripts/evaluate.py \
    --models runs/seed/final.pt runs/shard_0*/final.pt \
             runs/merged_123.pt runs/merged_unscaled.pt runs/avg.pt \
    --data data/fineweb/val.bin \
           data/fineweb/part_01.val.bin data/fineweb/part_02.val.bin \
    --batch-size 16 --out results/main.json
```

Prints a grid (one row per model, one column group per file: loss, perplexity, top-1
next-token accuracy) plus parameter counts and token provenance, and writes the full JSON.
Evaluation walks each file sequentially (no shuffling), so numbers are reproducible;
add `--fp32` for bit-stable numbers, `--max-batches N` for quick passes.

Qualitative samples: `--generate --tokenizer gpt2 --prompt "..."`.

What "success" looks like for SAP-Pure (no healing, by design): the merged model lands
*near* its parents on the global val (each parent saw only 1/N of D) while the
weight-average baseline collapses. Merged ≫ parents is not expected without healing.

---

## Running on multiple PCs (how asynchrony actually works)

There is no networking, no scheduler, no shared filesystem — coordination is entirely
through files. To train shard k on another machine, copy to it:

1. this repository (identical version — run `pytest` there to confirm),
2. `configs/<your-family>.json` — byte-identical on every machine,
3. `data/<corpus>/part_k.bin` + `data/<corpus>/meta.json` (and optionally `val.bin`),
4. `runs/seed/final.pt` — only if this shard branches from the seed.

Then run the shard's training command there. The trainer cross-checks the family config
against `meta.json`'s vocab size and refuses obvious mismatches. When the shard is done,
copy back one file: `runs/shard_k/final.pt` (it's stripped of optimizer state, so it's
the small one; `latest.pt` stays on the training machine in case you want to extend the
run later). Merge on any machine that has all the `final.pt`s.

Checklist for "will these merge later?": same family JSON, same tokenizer (enforced via
vocab check), same repo version. Widths, budgets, hardware, schedules may all differ —
that's the point.

The three commands per machine (data prep once, train N times anywhere, merge once) are
the entire distributed system.

## Scripts or notebooks?

Scripts. For multi-hour unattended training on a Linux box, `.py` + auto-resume is
strictly better than notebooks: a notebook dies with its kernel/SSH session and loses the
run; these scripts survive reboots by design. Run them detached:

```bash
tmux new -s shard01
python scripts/train.py ... 2>&1 | tee runs/shard_01/stdout.log
# detach: Ctrl+B then D; reattach later: tmux attach -t shard01
```

If a notebook is ever convenient (e.g. plotting `train_log.jsonl`, inspecting results
JSON, Colab demos), call the same CLIs from a cell (`!python scripts/train.py ...`) or
import the library (`from sap.merge import merge_checkpoints`). The logic lives in `sap/`
precisely so a notebook is just another thin caller — no logic is trapped in cells.

## What is verified, exactly

`python -m pytest tests -q` proves, on every machine you run it on:

- merged sublayer = α-weighted average of parents (pairwise, heterogeneous widths, random
  norm gains) — float64, ~1e-13;
- N-way merge with unequal alphas; sequential merge ≡ simultaneous merge weight-for-weight;
- over-complete merges (rectangular W_O, ΣH·d_head > d_model);
- unscaled mode = exact function sum;
- norm-gain absorption preserves the model's function;
- the exactness gate **fails** on a deliberately corrupted merge (it's not a rubber stamp);
- parameter accounting to the exact parameter: S_merged = E + Σ(Sᵢ−E) − (N−1)·(2L+1)·d_model
  (the last term: norm gains are counted once, not N times — a ~0.01% correction the
  thesis growth formula rounds away);
- checkpoint roundtrips; function-preserving seed growth; deterministic sampler resume;
- crash-resume reproduces the uninterrupted run's weights;
- end-to-end: train two shards → merge via files → token-weighted alphas correct → gate green.

See `explanation.md` for a function-by-function walkthrough of the entire codebase.
