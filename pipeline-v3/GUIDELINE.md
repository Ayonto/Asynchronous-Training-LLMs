# SAP — The Complete Usage Guideline

Start to finish: prepare a dataset once → train shards for as long as you want, on as
many PCs as you want → merge → evaluate. Written for the real long-running training
phase: budgets you can extend, graceful failures with reasons, checkpoint retention,
live progress, and sizing recipes for 500M / 1B targets.

---

## 0. The mental model (30 seconds)

| term | meaning |
|---|---|
| **family** | the fixed skeleton every model shares: `n_layers, d_model, d_head, vocab_size, max_seq_len, rope_theta, norm_eps` — one JSON file, decided once |
| **partition** | one slice of the dataset (`part_01.bin` …), made once at prep time |
| **shard** | one small model trained on one partition, fully independently |
| **merge** | tensor surgery that stacks all shards' heads + neurons into one wider model; gated by a built-in exactness check |
| **val sets** | `val.bin` = global held-out (every model compared on it); `part_XX.val.bin` = per-partition held-out (specialty checks) |

Key planning fact: **the merged big model is never trained** — it is assembled on CPU
and only evaluated. Training memory/time is always about the *shard* size.

## 1. One codebase, Linux and Windows

There is deliberately **one** codebase, not two folders. Everything is pure
Python + PyTorch with `pathlib` paths and atomic file operations that work identically
on both OSes — the same files are developed/tested on Windows and trained on Linux
every day. Two folders would drift apart and re-create the version-skew bugs we have
already been bitten by once.

- **Linux is the performance target**: bf16, fused kernels (when enabled), `tmux`.
- **Windows works everywhere it matters**: tests, data prep, small runs, progress
  viewing. Console output is plain ASCII so it renders on cp1252 terminals.
- Nothing in the code touches the operating system — no reboots, no cron, no services.
  The only system call anywhere is launching `python scripts/train.py` as a child
  process. Every failure the code can catch exits gracefully with the reason printed
  and recorded (honesty note: a kernel panic or power loss is below any program —
  no user-space code can prevent or survive the moment itself; what this code
  guarantees is that nothing is lost and the reason is visible afterwards).

Setup on every machine:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build (Linux box)
pip install -r requirements.txt
python -m pytest tests -q          # MUST print "26 passed" before any training
```

## 2. Prepare the dataset — ONCE (CPU job)

```bash
python scripts/prepare_data.py \
  --inputs hf:HuggingFaceFW/fineweb-edu:sample-10BT:train \
  --tokenizer gpt2 --tokenizer-backend tiktoken \
  --out-dir data/fineweb --num-partitions 5 \
  --val-fraction 0.005 --partition-val-fraction 0.001 \
  --max-tokens 40000000000
```

- `--tokenizer-backend tiktoken`: identical GPT-2 ids to HuggingFace, stable and fast.
- Partition sizing rule: each partition must hold at least the tokens its shard will
  train on (see sizing, §4). `--max-tokens` caps the total.
- Afterwards, verify: `python scripts/check_data.py --data-dir data/fineweb --decode 2`
  — must print ALL FILES CLEAN (every id inside the vocabulary, samples decode to
  real text).

Outputs: `part_XX.bin` (training partitions), `val.bin` (global held-out),
`part_XX.val.bin` (per-partition held-out), `meta.json` (tokenizer/vocab/dtype —
always travels with the bins).

## 3. Moving shards to other PCs (asynchronous training)

Tokenize once; never re-tokenize. To train shard k on another machine, copy:

1. this repository (same version; run `pytest` there — 26 passed);
2. the family JSON (byte-identical everywhere);
3. `part_k.bin` **and** `meta.json` (optionally `val.bin` for live val evals);
4. the seed checkpoint only if branching from a seed.

**The tokenizer does not travel** — training reads pre-tokenized ids. It is needed only
at prep time and for text generation during evaluation (fetched by name).

When a shard finishes, carry back ONE file: `runs/shard_k/final.pt`. Merge on any
machine that has all the finals.

## 4. Choosing shard sizes (the 500M / 1B recipes)

Free per shard: `--n-heads` (H) and `--d-ff`. Everything else is family-fixed. The
arithmetic (d_head=64, vocab 50257, reference family L=24, d_model=1536):

```
E (embedding, shared, counted once)     = vocab x d_model            = 77.2M
attention per layer                     = 4 x d_model x H x 64
ffn per layer (SwiGLU)                  = 3 x d_model x d_ff
shard  ≈ E + L x (attn + ffn)
merged ≈ E + N x L x (attn + ffn)       <- embedding does NOT multiply by N
```

Don't do this by hand — use the calculator:

```bash
# solve d_ff for a target merged size:
python scripts/plan_model.py --family configs/family_reference.json \
    --n-heads 8 --num-shards 5 --target-merged 1000000000
# or check a configuration you have in mind:
python scripts/plan_model.py --family configs/family_reference.json \
    --n-heads 4 --d-ff 448 --num-shards 5
```

Ready-made recipes on `configs/family_reference.json`, N = 5 shards:

| target merged | per-shard flags | shard size | merged size | Chinchilla tokens/shard | ckpt size (w/ optimizer) |
|---|---|---|---|---|---|
| **~0.5B** | `--n-heads 4 --d-ff 448` | 165M | **514M** | ~3.3B | ~2.0 GB |
| **~1B** | `--n-heads 8 --d-ff 1024` | 266M | **1.02B** | ~5.3B | ~3.2 GB |

(The validation family `configs/family_validation.json` with `--n-heads 4 --d-ff 768`
gives 35M shards → ~96M merged — the small dress rehearsal.)

Also from the calculator: static GPU memory (weights+grads+AdamW) and disk per shard
with checkpoint retention. Rule of thumb for the 4090: both recipes train comfortably;
tune `--batch-size` per §8.

## 5. Every training setting, in plain language

`python scripts/train.py --help` shows them all; here is what they mean:

| flag | plain meaning |
|---|---|
| `--name` | label for logs/checkpoints (e.g. `shard_03`) |
| `--family` | the family JSON — same file on every machine, forever |
| `--data` / `--val` | the training partition .bin / optional held-out .bin for live val scores |
| `--out-dir` | where checkpoints, logs, and status land (one folder per shard) |
| `--n-heads`, `--d-ff` | the shard's width (its size) — from §4; omit with `--init-from` to inherit |
| `--init-from` | `scratch`, or a seed checkpoint path to branch from |
| `--init-seed` | random-init seed; give every from-scratch shard a DIFFERENT one |
| `--batch-size` | sequences processed at once — **the GPU-memory dial** (§8) |
| `--grad-accum` | batches accumulated before one weight update; raises effective batch without memory |
| `--seq-len` | **context window**: how many tokens of text the model sees per training example (and the length it learns to handle). Longer = more memory and slower steps. Must be ≤ family `max_seq_len`; keep it the same for all shards. 1024 is our standard. |
| `--lr`, `--min-lr`, `--warmup-steps` | learning-rate peak, floor, and ramp-up steps |
| `--schedule` | `cosine` (decays to min-lr over the budget; needs a known horizon) or `constant`. **For open-ended "keep improving" training use `constant`** — cosine's shape depends on the configured end point, so extending a cosine run changes future LRs. |
| `--weight-decay`, `--grad-clip` | standard regularization / gradient-spike protection; defaults are fine |
| `--data-seed` | shuffling seed; determines the exact (reproducible) batch order |
| `--max-epochs/-hours/-steps` | ABSOLUTE budget; first one hit stops the run and writes `final.pt` |
| `--extend-hours/-epochs` | RELATIVE budget: train this much MORE on top of whatever is done (§6) |
| `--checkpoint-every-min/-steps` | steady checkpoint cadence cap (a backoff ramp handles the early phase) |
| `--keep-prev` | how many previous checkpoint generations to retain (§7); default 3 |
| `--fresh` | deliberately restart from step 0, ignoring existing checkpoints |
| `--log-interval`, `--eval-interval`, `--eval-batches` | console/status cadence and live val evals |
| `--device/--dtype/--compile/--sdpa/--no-pin-memory/--no-fused-adamw` | hardware knobs; see the stability profile (§8) |

## 6. Training a shard for a long time (budgets, extending, stopping, failing)

The standard long-run command (per shard, per PC):

```bash
python scripts/train.py --name shard_01 --family configs/family_reference.json \
  --data data/fineweb/part_01.bin --val data/fineweb/val.bin \
  --out-dir runs/shard_01 --n-heads 8 --d-ff 1024 --init-seed 1001 \
  --batch-size 8 --grad-accum 16 --seq-len 1024 --schedule constant \
  --max-epochs 1
```

**Crash / interruption → resume.** Rerun the *identical* command. It finds the newest
usable checkpoint automatically (even if the crash hit in the middle of a checkpoint
save — see §7) and continues with identical optimizer state and data order
(test-verified byte-equivalent to an uninterrupted run).

**Keep improving after the budget is over.** Budgets are not walls; extend anytime:

```bash
# after --max-epochs 1 completed, train ONE MORE epoch:
python scripts/train.py ... --extend-epochs 1
# or: train 6 more hours on top of whatever has been done:
python scripts/train.py ... --extend-hours 6
```

`--extend-*` means "MORE, from where it stands" — no cumulative arithmetic. Each
completion rewrites `final.pt`, so merge/eval always see the newest finished state.
(With `--schedule constant` extension is seamless; that's why it's recommended for
open-ended runs.)

**Stop a running shard gracefully.** `Ctrl+C` (or `kill <pid>` on Linux — the pid is in
`status.json`): the shard finishes its current step, saves a checkpoint, records
`stopped` in its status, and exits cleanly. It does NOT write `final.pt` (it's paused,
not finished) — resume or extend later with the same command.

**Failures are graceful and explained.** Any error the process can catch (OOM, bad
data, disk full, …) triggers: emergency checkpoint → reason written to `status.json`
and `train_log.jsonl` → a clear block on the terminal:

```
[shard_01] TRAINING FAILED
  reason : OutOfMemoryError: ... (fix: lower --batch-size and raise --grad-accum)
  state  : progress saved to runs/shard_01/latest.pt
  resume : rerun the same command — it continues from the last checkpoint
```

A segfault/power-cut can't print (the process is killed by the OS), but the retained
checkpoints (§7) mean nothing is lost, and the progress viewer marks the shard
`stale?` so you notice.

## 7. Checkpoints: what is stored, retention, disk math

Per shard folder:

| file | contents |
|---|---|
| `latest.pt` | newest resumable state (model + optimizer + step/tokens/RNG/data position) |
| `prev_1.pt` … `prev_3.pt` | the previous `--keep-prev` generations (prev_1 = newest). Rotation is by RENAMES — each save serializes the big file exactly once. |
| `final.pt` | written only when a budget completes; optimizer stripped (≈⅓ the size) — the portable artifact for merge/eval |
| `status.json` | tiny live-progress snapshot (§9) |
| `train_log.jsonl` | one JSON line per log interval — plot loss curves from this |

So: **not only the latest** — the previous 3 generations are kept for safety, capped
(oldest is deleted), and the save order is crash-safe at every instant: the new file is
fully written *before* any rotation, so whatever moment a crash hits, a complete
checkpoint exists and resume finds it (this exact scenario is unit-tested, including
"crash mid-rotation" and "crash between rotation and rename").

Disk per shard ≈ `(1 + keep_prev) x ckpt_size + final.pt` — e.g. the 1B recipe:
4 × 3.2 GB + 1.1 GB ≈ **14 GB**. Lower `--keep-prev` if disk is tight.

## 8. Memory & hardware knobs (any GPU)

```
tokens per optimizer step = batch_size x grad_accum x seq_len     (the math)
peak GPU memory           = f(batch_size, seq_len, shard width)    (the hardware)
```

**Rule: fit `batch_size` to the card, compensate with `grad_accum`** (same math,
different memory). Leave ~40–50% VRAM headroom — with a 50k vocab each forward
materializes a `(batch, seq, vocab)` logits tensor, and long runs fragment the
allocator; running at 90% is how you OOM at hour 3. Watch `nvidia-smi` for the first
10 minutes and adjust.

Starting points (seq 1024, bf16): 24 GB → `8 x 16` (266M shard) or `12 x 12`
(165M shard); 32 GB (5090) → roughly 1.5×; 48 GB (A6000) → roughly 2–3× the batch.

**Stability profile** (all optional native fast-paths off; identical math, ~2× slower —
what `validation/validate.py` uses after this box's crash history):
`--dtype fp32 --sdpa math --no-fused-adamw --no-pin-memory` and no `--compile`.
Once a machine proves itself over full nights, re-enable ONE per run in this order for
speed: `--dtype auto` (bf16) → `--sdpa auto` (flash attention) → fused adam → pinned
memory → `--compile`.

## 9. Watching progress (lightweight, no terminal spam)

Every shard refreshes a tiny `status.json` (atomic, every log interval — zero training
overhead). View all shards on a machine in one table, without touching training:

```bash
python scripts/progress.py --runs-dir runs              # one snapshot, exits
python scripts/progress.py --runs-dir runs --watch 30   # live table, refresh 30s
```

```
shard       state       step  ep  tokens   loss  val_ppl  hours   prog  ETA(h)  updated
shard_01    running   12,340   0    1.6B  3.412    31.2    6.21  51.7%     5.8  12s ago
shard_02    completed 24,000   1    3.1B  3.105    24.9   12.02 100.0%     0.0  3m ago
shard_03    failed         …                                              (reason shown below table)
```

`stale?` = marked running but silent >5 min → the process died without cleanup (e.g.
segfault); rerun its command to resume. The training terminal itself stays as quiet as
you set `--log-interval`.

## 10. Merge and evaluate (unchanged)

```bash
python scripts/merge.py --inputs runs/shard_0*/final.pt --out runs/merged.pt            # exact average (default, token-weighted)
python scripts/merge.py --inputs ... --out runs/merged_uns.pt --no-scale                # exact sum
python scripts/merge.py --inputs ... --out runs/avg.pt --method avg --alpha-mode uniform # naive baseline
python scripts/evaluate.py --models runs/shard_0*/final.pt runs/merged.pt \
    --data data/fineweb/val.bin data/fineweb/part_0*.val.bin --out results/grid.json
```

Every stack merge must pass its exactness gate or it refuses to save. Evaluate on
`val.bin` for the headline comparison; per-partition vals for specialty retention.

## 11. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `TRAINING FAILED / OutOfMemoryError` | batch too big for the card | lower `--batch-size`, raise `--grad-accum`; rerun (auto-resumes) |
| shard shows `stale?` in progress viewer | process died without cleanup (segfault/kill) | rerun the same command — resumes from newest checkpoint |
| `resume mismatch on '...'` | a resume-critical setting changed mid-run | intended guard; restore the setting, or `--fresh` to restart deliberately |
| training refuses: "token id ≥ vocab_size" | corrupt .bin or wrong tokenizer | re-prep; verify with `scripts/check_data.py` |
| segfault during tokenization | HF Rust tokenizer | use `--tokenizer-backend tiktoken` |
| random segfaults / whole-PC crashes | below user-space: RAM / driver / power | `python diagnose.py`; `dmesg -T \| tail -50` after a crash names the component; stability profile (§8) minimizes exposure |
| cosine LR "warning: needs a known horizon" | only `--max-hours`/`--extend-hours` given | expected — it falls back to constant; or use `--schedule constant` explicitly |

## 12. The fixed-window orchestrator (validation runs)

`python validation/validate.py` remains the "fit everything in one 12h window on one
PC" tool (auto-chains shards in isolated subprocesses with retries, then merges and
evaluates). The real long-run training above doesn't need it — each shard is just its
own `scripts/train.py` command, which IS the asynchronous workflow. The progress
viewer works on `validation/runs` too.
