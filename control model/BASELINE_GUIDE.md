# Baseline Guide — training the conventional control model

The merged model has no meaning on its own. "546M merged model reaches perplexity X"
answers nothing until you can say **what a normal 546M model trained on the same data
reaches**. This guide builds that control.

This is baseline #4 in the framework report (§13.3) — *"the full-data model at equal
total compute"*, described there as **the real target**. It is the single most important
missing piece of the experiment, and everything else in the thesis is measured against it.

---

## 0. The mental model

| | merged model | baseline model |
|---|---|---|
| how it was made | 5 shards trained separately on `part_01..05`, then stacked | one run over `part_all.bin` |
| architecture | 5× heads, 5× d_ff per layer | **identical** (copied from the merged checkpoint) |
| data seen | union of all partitions | the same union |
| training | 5 independent runs, zero communication | one conventional run |

Only the **training paradigm** differs. That is the whole experiment.

Two scripts, both independent of the merge pipeline (neither imports `sap.merge`):

```
scripts/combine_partitions.py   part_01..05.bin  ->  part_all.bin
scripts/train_baseline.py       part_all.bin     ->  runs/baseline/final.pt
```

---

## 1. Combine the partitions

The shards each saw one partition. The control must see all of them.

```bash
python scripts/combine_partitions.py --data-dir data/fineweb_val
```

```powershell
python scripts\combine_partitions.py --data-dir data\fineweb_val
```

Output: `data/fineweb_val/part_all.bin` plus `part_all.manifest.json` recording exactly
which files went in, in what order, at what offsets.

**What it includes:** every `part_NN.bin`.
**What it deliberately excludes:**

| file | why excluded |
|---|---|
| `val.bin` | global held-out — no model may ever train on it |
| `part_XX.val.bin` | per-partition held-out — same reason |
| `seed.bin` | it is a *sample of* the partitions, so its documents are already in `part_*.bin`. Including it would silently upweight those documents and break the token match. |

**Why plain concatenation is enough.** `ChunkSampler` draws a fresh random permutation of
all chunks every epoch, so byte order in the file has no effect on the batch sequence.
There is nothing to gain from interleaving.

Your existing `meta.json` is left untouched and keeps working — the trainer reads dtype
and vocab from it automatically because the new file sits in the same directory.

> **Disk:** `part_all.bin` is the sum of the partitions (~2× your dataset on disk in
> total). If space is tight, note that the partitions are still needed for the shards,
> so budget for both.

---

## 2. Train the baseline

```bash
python scripts/train_baseline.py \
  --name baseline_546m \
  --family configs/family_reference.json \
  --data data/fineweb_val/part_all.bin \
  --val  data/fineweb_val/val.bin \
  --out-dir runs/baseline_546m \
  --match-merged runs/merged/merged.pt \
  --match-tokens-from runs/shard_01/final.pt runs/shard_02/final.pt \
                      runs/shard_03/final.pt runs/shard_04/final.pt \
                      runs/shard_05/final.pt \
  --batch-size 4 --grad-accum 16 \
  --checkpoint-every-min 30
```

```powershell
python scripts\train_baseline.py `
  --name baseline_546m `
  --family configs\family_reference.json `
  --data data\fineweb_val\part_all.bin `
  --val  data\fineweb_val\val.bin `
  --out-dir runs\baseline_546m `
  --match-merged runs\merged\merged.pt `
  --match-tokens-from runs\shard_01\final.pt runs\shard_02\final.pt `
                      runs\shard_03\final.pt runs\shard_04\final.pt `
                      runs\shard_05\final.pt `
  --batch-size 4 --grad-accum 16 `
  --checkpoint-every-min 30
```

**Always dry-run first.** Add `--dry-run` to print the resolved plan — widths, parameter
count, token target, step count, memory estimate — and exit without training. Read it
before committing GPU-days.

### The two matching flags

`--match-merged runs/merged/merged.pt`
Reads the merged checkpoint's per-layer `(n_heads, d_ff)` and builds a from-scratch model
with exactly those widths. Nothing is hardcoded, so this stays correct if you change the
shard count or shard size. It then cross-checks its computed parameter count against the
merged checkpoint's actual tensors and prints `exact match` or a diff.

`--match-tokens-from <shard checkpoints>`
Sums `meta.tokens_seen` over the shards and converts it into `--max-steps`. This handles
multi-epoch shards automatically: if each shard did 2 epochs, the sum is 2×|D| and the
baseline is given the same.

Alternatives: `--match-shards <ckpts>` computes the merged widths from the shards without
needing the merged file; `--n-heads`/`--d-ff` set widths explicitly.

---

## 3. Controlling the budget

Three limits; **whichever is hit first stops the run**. Mix freely.

| flag | meaning |
|---|---|
| `--max-epochs N` | N full passes over `part_all.bin` |
| `--max-hours H` | H hours of **cumulative** wall time across all resumes |
| `--max-steps S` | S optimizer steps |
| `--match-tokens T` / `--match-tokens-from` | converted to `--max-steps` for you |

Practical patterns:

```bash
# "run for two days, whatever it reaches"
--max-hours 48

# "exactly one pass over all the data"
--max-epochs 1

# "exactly as many tokens as the shards collectively saw"  <- the fair comparison
--match-tokens-from runs/shard_*/final.pt

# "overnight, but stop early if it finishes the epoch"
--max-hours 10 --max-epochs 1
```

> **Cosine schedule needs a horizon.** With `--schedule cosine` (the default) the LR decay
> is computed from `max_steps` or `max_epochs`. If you give only `--max-hours`, the
> trainer warns and falls back to constant LR after warmup. For a clean run, give a step
> or epoch target as well — or use `--schedule constant` deliberately.

---

## 4. Making the comparison fair — read this before reporting anything

There are two defensible comparisons and they answer **different questions**. Report both.

**Equal tokens** — *"is merging as good as one conventional run over the same data?"*
Baseline gets `--match-tokens-from` the shards. Same data, same token count, same
architecture. This is the headline number.

**Equal wall-clock / equal single-GPU compute** — *"what does a researcher with one GPU
and T hours get?"*
Baseline gets `--max-hours T` where T is *one shard's* budget, not the sum. This is the
comparison that makes the paradigm look good, because the merged model consumed 5× the
GPU-hours — spread across machines that never had to talk to each other. That is the
actual selling point of the thesis, and it deserves its own row rather than being
smuggled into the equal-token row.

Be explicit about which one a given number is. The most common way merging papers get
attacked is by quietly comparing N× the compute against 1×.

| | tokens | GPU-hours | parameters |
|---|---|---|---|
| one shard | \|D\|/5 | T | 171M |
| merged (5 shards) | \|D\| | 5T (parallelizable, no interconnect) | 546M |
| baseline, equal tokens | \|D\| | ~5T (one machine) | 546M |
| baseline, equal wall-clock | \|D\|/5 | T | 546M |

---

## 5. Checkpointing and crash recovery

Inherited unchanged from `sap/train.py`, which was already built for an unreliable PC.

* Checkpoints are written **atomically** (temp file + rename), so a power cut mid-save
  cannot corrupt the previous checkpoint.
* `latest.pt` stores model, optimizer, step, token count, cumulative wall time, RNG
  states, and the sampler's `(epoch, cursor)` position.
* **To resume, rerun the exact same command.** It picks up `latest.pt` automatically and
  reproduces the batch sequence the uninterrupted run would have seen.
* Checkpoint cadence ramps: the first save lands ~60s into every session, then the
  interval doubles up to `--checkpoint-every-min`. A machine that keeps crashing after
  ten minutes still banks progress on every attempt; a stable multi-day run performs only
  a handful of writes.

```bash
# after a crash — identical command, nothing else to do
python scripts/train_baseline.py --name baseline_546m ... (same flags)

# extend a finished run: raise the budget, rerun
python scripts/train_baseline.py ... --max-hours 96

# deliberately start over
python scripts/train_baseline.py ... --fresh
```

A resume **refuses to proceed** if you changed `data_path`, `batch_size`, `grad_accum`,
`seq_len`, `data_seed`, the widths, or `init_from`, because that would silently corrupt
the data and budget bookkeeping. Change them only with `--fresh`.

`--keep-history 2` (the default here) retains two extra step-stamped snapshots alongside
`latest.pt`, so a checkpoint that turns out to be bad is not the only one you have.

---

## 6. Memory

**This is a new regime for your machine.** The GUIDELINE's claim that "the merged big
model is never trained" is true of the merge pipeline — but the baseline *is* trained at
merged size. Budget accordingly:

| | ~546M model |
|---|---|
| params + grads + Adam moments (fp32) | ~8.7 GB |
| activations | scales with `batch_size × seq_len` |
| realistic minimum | 16 GB, comfortable on 24 GB |

The dry-run prints the optimizer-state estimate for your actual configuration.

**If you OOM:** halve `--batch-size` and double `--grad-accum`. The optimizer batch — and
therefore the training math — is unchanged; only the memory peak drops.

```bash
--batch-size 8 --grad-accum 8     # \
--batch-size 4 --grad-accum 16    #  |- all the same effective batch of 64
--batch-size 2 --grad-accum 32    # /
```

Other levers, in order of preference: `--seq-len` below the family's `max_seq_len`
(changes what the model learns, so keep it equal to the shards'), `--dtype bf16` (the
default on capable GPUs), `--no-pin-memory` and `--sdpa math` on flaky driver stacks.

---

## 7. Verification checklist

Before you trust a baseline number:

- [ ] `python -m pytest tests -q` prints **20 passed** on this machine.
- [ ] `--dry-run` reported `-> exact match: the control is an architectural twin.`
- [ ] `part_all.manifest.json` lists every partition, and its `total_tokens` equals the
      sum of the partition token counts in `data/<set>/meta.json`.
- [ ] The baseline's `meta.tokens_seen` is within ~1% of the shards' summed
      `tokens_seen` (the ceiling in the step conversion causes a fraction of a step of
      slack).
- [ ] The baseline never saw `val.bin` — confirm `--data` points at `part_all.bin`.
- [ ] Both models are evaluated with the same `--seq-len` and `--batch-size`.

Then:

```bash
python scripts/benchmark.py \
  --models runs/merged/merged.pt runs/baseline_546m/final.pt \
  --bins data/fineweb_val/val.bin \
  --out runs/benchmark
```

---

## 8. What to expect

Be prepared for the baseline to **win**, and plan the thesis around that being an
acceptable outcome. SAP-Pure is structurally an ensemble folded into one network: its
per-sublayer output is a convex average of the parents, so it is bounded above by the
ensemble of its shards, and the composition gap (report §4.4, §12 Risk 1) costs
something on top. A conventionally trained model has no such ceiling.

The thesis claim that survives either result is the one in §12: **structural merging
degrades gracefully where parameter averaging collapses**, at a fraction of the
coordination cost. Frame the baseline as the reference point, not as a contest the
merged model must win. The interesting numbers are the *gap* and what closes it —
the healing pass (§6.5b), the branch point, the number of shards.

Include the weight-averaging baseline in the same table (`scripts/merge.py --method avg`).
If stack-merging sits between weight-averaging and the conventional baseline, that is a
clean, publishable result.

---

## 9. Troubleshooting

| symptom | cause / fix |
|---|---|
| `no meta.json found next to <bin>` | run `combine_partitions.py` into the dataset directory, or pass `--data-dtype uint16` |
| `resume mismatch on 'batch_size'` | you changed a resume-critical flag; restore it or pass `--fresh` |
| `dataset too small: N chunks ... but batch_size is B` | `--seq-len × --batch-size` exceeds the file; lower one |
| `training from scratch requires --n-heads and --d-ff` | you passed none of `--match-merged` / `--match-shards` / `--n-heads` |
| widths differ from the merged model | you used explicit `--n-heads`; prefer `--match-merged` |
| CUDA OOM at step 0 | see §6 — batch/accum trade |
| OOM *after* an eval | the trainer already calls `empty_cache()` after evals; lower `--eval-batches` |
| loss is NaN | lower `--lr`; confirm `--grad-clip 1.0`; try `--dtype fp32` to isolate |
| segfault mid-run | `--sdpa math --no-pin-memory` |
| baseline much worse than a single shard | check `tokens_seen` — the token match may have produced far fewer steps than you expected |

---

## 10. Command reference

```bash
python scripts/combine_partitions.py --data-dir data/fineweb_val
python scripts/combine_partitions.py --inputs a.bin b.bin --out c.bin --force
python scripts/train_baseline.py --help
python scripts/train_baseline.py ... --dry-run
```

Next: [FINETUNE_GUIDE.md](FINETUNE_GUIDE.md) — fine-tuning both models for downstream
evaluation, manual generation, and the benchmark harness.
