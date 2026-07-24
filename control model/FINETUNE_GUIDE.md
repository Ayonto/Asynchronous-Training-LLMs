# Fine-tuning, Generation and Benchmarking Guide

Covers three scripts:

| script | purpose |
|---|---|
| `scripts/finetune.py` | fine-tune any checkpoint for text classification |
| `scripts/generate.py` | type a prompt, read what the model says |
| `scripts/benchmark.py` | one consolidated merged-vs-baseline report |

Prerequisite: a merged model and a conventionally trained baseline. See
[BASELINE_GUIDE.md](BASELINE_GUIDE.md).

```bash
pip install datasets       # needed for the built-in HuggingFace datasets
```

---

## 1. Why fine-tune at all

You are right that a pretrained-only model cannot be benchmarked meaningfully. Two
reasons, and it is worth stating both in the thesis:

1. **Perplexity is a weak single instrument.** It is sensitive to tokenizer and corpus,
   it is not comparable across vocabularies, and it says nothing about whether the
   representations are *usable*. Two models with the same perplexity can differ
   substantially downstream.
2. **Zero-shot benchmarks do not work at this scale.** MMLU, HellaSwag and ARC are
   near-chance for a ~500M model trained on a few billion tokens. Reporting 25.3% vs
   25.9% on MMLU is reporting noise. Fine-tuned classification is the standard,
   defensible instrument in this size class.

There is also a merge-specific reason. The composition gap (report §4.4) predicts that
the merged model's deep layers read a blended residual stream neither parent trained on.
If that damages the representations, **fine-tuning is exactly where it shows up** — the
merged model would need more steps or reach a lower plateau than the baseline. Perplexity
alone can hide this; a linear probe cannot.

---

## 2. Which dataset — recommendation

### Primary: **AG News** (`--dataset ag_news`)

News topic classification: World / Sports / Business / Sci-Tech. 120k train, 7.6k test,
4 balanced classes.

Why this one:

* **Large enough to be low-variance.** 7.6k test examples means a 1% difference is not
  seed noise. This matters more than anything else — SST-2's 872-example validation set
  makes small differences unresolvable.
* **4-way, balanced.** Chance is 25%, so the dynamic range is wide and accuracy is
  directly interpretable. Binary tasks compress differences near the ceiling.
* **It tests what pretraining actually gave you.** Topic classification depends on world
  knowledge and vocabulary — precisely what FineWeb-Edu pretraining confers, and
  precisely what merging might damage. A syntax-only task would not discriminate.
* **Short documents.** 256 tokens covers most examples, so runs are fast and
  `--max-length` truncation is not a confound.
* Ubiquitous in the literature, so your numbers are legible to a reader.

### Secondary: **SST-2** (`--dataset sst2`)

Binary sentiment on short sentences, 67k train. Tests something genuinely different
(sentiment, not topic). Its 872-example validation split is small — use **5 seeds** and
expect wider error bars. Good as a second data point, poor as a headline.

### Also available

| name | classes | note |
|---|---|---|
| `imdb` | 2 | long documents; use `--max-length 512` to test long-context retention |
| `dbpedia_14` | 14 | 560k train, saturates above 98% — a sanity check, not a discriminator |
| `trec` | 6 | 5.5k train; low-resource stress test, high variance |

`python scripts/finetune.py --list-datasets` prints this with the full rationale.

**Suggested plan:** AG News as the headline (3 seeds, full fine-tune **and** linear
probe), SST-2 as corroboration (5 seeds, full fine-tune). If they agree, you have a
result. If they disagree, that itself is worth reporting.

---

## 3. Running it

Identical flags for both models — that is the point.

```bash
# merged
python scripts/finetune.py \
  --init-from runs/merged/merged.pt --name merged \
  --dataset ag_news --out-root runs/finetune \
  --seeds 0 1 2 --max-epochs 3 --batch-size 16 --max-length 256

# baseline
python scripts/finetune.py \
  --init-from runs/baseline_546m/final.pt --name baseline \
  --dataset ag_news --out-root runs/finetune \
  --seeds 0 1 2 --max-epochs 3 --batch-size 16 --max-length 256
```

```powershell
python scripts\finetune.py `
  --init-from runs\merged\merged.pt --name merged `
  --dataset ag_news --out-root runs\finetune `
  --seeds 0 1 2 --max-epochs 3 --batch-size 16 --max-length 256
```

Also fine-tune a **single shard** (`runs/shard_01/final.pt`) as a lower bound — it saw
1/5 of the data, so it shows how much merging actually recovered.

Outputs per run: `runs/finetune/<name>_<dataset>_seed<K>/` with `best.pt`, `final.pt`,
`summary.json`, `finetune_log.jsonl`; plus `<name>_<dataset>_aggregate.json` with the
across-seed mean ± std.

### Always run both modes

**Full fine-tune** (default) — the headline number, but it confounds representation
quality with adaptability: enough gradient steps can paper over a weak backbone.

**Linear probe** (`--freeze-backbone`) — trains only the classification head, so it
measures the frozen representations directly. Cheap, fast, and far more sensitive to
merge damage. If the merged model's probe accuracy is below the baseline's while their
full fine-tunes are level, you have located the composition gap. That is a *result*, not
a problem.

```bash
python scripts/finetune.py \
  --init-from runs/merged/merged.pt --name merged_probe \
  --dataset ag_news --out-root runs/finetune \
  --seeds 0 1 2 --freeze-backbone --head-lr 1e-3 --max-epochs 5
```

### Seeds are not optional

Fine-tuning variance at this scale is comfortably larger than the effects you are
measuring. **A single-seed difference of one or two points is not evidence.** Use at
least 3 seeds (5 on SST-2); `--seeds 0 1 2` runs them in one command and prints
mean ± std. `benchmark.py` then reports a confidence interval on the *difference*.

---

## 4. Changing the dataset

Yes — three ways, no code changes.

**Built-in name:**
```bash
--dataset sst2
```

**Any HuggingFace classification dataset:**
```bash
--dataset hf:yelp_polarity --num-labels 2 --text-key text --label-key label
--dataset hf:glue:cola --num-labels 2 --text-key sentence --label-key label \
  --eval-split validation
```

**Local files** (`.jsonl` or `.csv`):
```bash
--dataset file:mydata/train.jsonl --eval-file mydata/test.jsonl \
  --num-labels 3 --text-key text --label-key label
```
`.jsonl` = one object per line, e.g. `{"text": "...", "label": 2}`. Labels must be
integers `0..K-1`.

To add a permanent entry, append a `DatasetSpec` to the `DATASETS` dict in
`sap/finetune.py` — it then works by name and appears in `--list-datasets`.

**Always re-run every model on a new dataset.** Cross-dataset numbers are not comparable.

---

## 5. How the classifier works

```
input_ids -> embed -> L blocks -> final_norm -> [pool] -> dropout -> Linear(d_model, C)
```

The backbone is an unmodified `SAPModel`; the merge math is untouched. Design decisions
worth knowing, because a committee may ask:

* **Pooling at the last real token.** A causal LM only "sees" the whole sequence at its
  final position, so that is where the sequence representation lives. This is what
  `GPT2ForSequenceClassification` does.
* **Right padding, no attention mask.** Attention is causal, so position *t* attends only
  to positions ≤ *t*. Pad tokens sit strictly to the right of the last real token and
  therefore cannot influence the pooled position. Adding a padding mask would change
  nothing.
* **Tail truncation.** Long documents keep their **last** `--max-length` tokens, since
  those are nearest the pooling position.
* **Two learning rates.** The pretrained backbone gets `--lr 2e-5`; the randomly
  initialized head gets `--head-lr 1e-3`. One shared LR either destroys the backbone or
  leaves the head untrained.
* **Shared tokenization cache.** Both models read the same `.npz`, so they are trained
  and evaluated on byte-identical inputs. Delete `data/finetune_cache/` to rebuild.

> The `--tokenizer` **must** match pretraining (`gpt2` here). A mismatch silently
> produces garbage rather than an error.

### Hyperparameters

| flag | default | notes |
|---|---|---|
| `--lr` | 2e-5 | backbone; 1e-5–5e-5 is the usable band |
| `--head-lr` | 1e-3 | raise to 1e-2 for a linear probe |
| `--max-epochs` | 3 | 2–4 for AG News; more overfits |
| `--batch-size` | 16 | pair with `--grad-accum` if memory is tight |
| `--max-length` | 256 | 512 for IMDB |
| `--dropout` | 0.1 | |
| `--max-train` | all | subsample for smoke tests / low-resource curves |

Tune on **one** model, then use identical values everywhere. Tuning per-model makes the
comparison meaningless.

### Checkpointing

Same contract as pretraining: atomic writes, `--checkpoint-every-min` (default 20),
resume by rerunning the same command, `--fresh` to restart. `best.pt` tracks the best
eval accuracy; `final.pt` is the end state. Resume refuses if you changed `init_from`,
`dataset`, `max_length`, `batch_size`, `grad_accum`, `seed`, or `freeze_backbone`.

---

## 6. Generation — the manual smell test

```bash
# interactive
python scripts/generate.py --model runs/merged/merged.pt

# one-shot
python scripts/generate.py --model runs/merged/merged.pt \
  --prompt "The three laws of thermodynamics state that" --max-new-tokens 120

# both models on the same prompt and seed, side by side
python scripts/generate.py \
  --model runs/merged/merged.pt --compare runs/baseline_546m/final.pt \
  --prompt "Photosynthesis is the process by which" --seed 0

# deterministic
python scripts/generate.py --model runs/merged/merged.pt \
  --prompt "In 1969, the Apollo 11 mission" --greedy

# batch
python scripts/generate.py --model runs/merged/merged.pt --prompt-file prompts.txt
```

Works on any checkpoint: shard, merged, baseline, or a fine-tuned classifier (its
backbone is extracted automatically). Output streams token by token.

Interactive commands: `/temp 0.7` `/topk 50` `/topp 0.9` `/rep 1.1` `/len 200`
`/seed 123` `/greedy` `/params` `/quit`.

Sampling supports top-k, top-p and a repetition penalty — implemented in the script, so
`sap/model.py` stays untouched.

> **Speed.** There is no KV cache: every new token re-runs the full forward pass, so
> generation is O(n²). Fine for 100–200 tokens on a GPU; slow on CPU at 500M. Use
> `--device cuda`.

**What to look for.** Use `--compare` with a fixed `--seed` and read them side by side.
Degeneration specific to merging — repetition loops, topic drift mid-sentence, confident
nonsense — is a qualitative signature of the composition gap, and a paragraph of real
samples in the thesis is worth more than another table.

---

## 7. Benchmarking

```bash
# perplexity only — available the moment the baseline finishes pretraining
python scripts/benchmark.py \
  --models runs/merged/merged.pt runs/baseline_546m/final.pt \
  --bins data/fineweb_val/val.bin \
  --out runs/benchmark

# the full report
python scripts/benchmark.py \
  --models runs/merged/merged.pt runs/baseline_546m/final.pt \
           runs/shard_01/final.pt runs/weightavg/avg.pt \
  --bins data/fineweb_val/val.bin \
         data/fineweb_val/part_01.val.bin data/fineweb_val/part_02.val.bin \
  --finetune-root runs/finetune \
  --compare merged baseline \
  --out runs/benchmark
```

Three sections, all optional depending on what you pass:

1. **Language modelling** — loss / perplexity / top-1 accuracy per model per file. The
   per-partition columns are the specialty-retention check: a merged model should be
   competitive on *every* `part_XX.val.bin`, while each shard is strong only on its own.
2. **Downstream** — fine-tuned accuracy and macro-F1, aggregated over seeds, with a
   **95% confidence interval on the difference** (Welch, unequal variances). It prints
   `RESOLVED` or `NOT RESOLVED` depending on whether the interval excludes zero. A CI is
   reported rather than a p-value because the effect size and its uncertainty are what a
   reader needs.
3. **Accounting** — parameters and pretraining tokens per model, so an unfair comparison
   is visible in the same table as the result.

Everything lands in `runs/benchmark/benchmark_report.json`.

---

## 8. Which benchmarks to use, and what to expect

**Tier 1 — do these. They work at 500M.**

| benchmark | why |
|---|---|
| Held-out perplexity on `val.bin` | primary LM metric; already implemented |
| Per-partition perplexity | specialty retention — a merging-specific result nobody else reports |
| AG News fine-tune (3 seeds) | headline downstream number |
| AG News linear probe (3 seeds) | representation quality without the fine-tuning confound |
| SST-2 fine-tune (5 seeds) | corroboration on a different task type |

**Tier 2 — worth trying, may show signal.**

Via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), which
needs a small HuggingFace-format adapter around `SAPModel`:

* **LAMBADA** — last-word prediction. The most informative zero-shot task at this scale;
  it needs long-range coherence, exactly what the composition gap would damage.
* **PIQA**, **ARC-easy** — sometimes above chance at 500M.
* **BLiMP** — targeted syntactic minimal pairs; small models do respectably, and it
  isolates grammar from knowledge.

**Tier 3 — do not bother.** MMLU, HellaSwag, ARC-challenge, GSM8K, TruthfulQA, anything
generative-graded. At 500M with a few billion tokens these are at or near chance, and
reporting them invites the question of why you did.

**Sample-efficiency curve — the strongest optional addition.** Fine-tune both models on
1%, 10%, 50%, 100% of AG News (`--max-train 1200 12000 60000` etc.) and plot accuracy vs
training-set size. If the merged model needs more labelled data to reach the same
accuracy, that is the composition gap measured in a directly interpretable unit, and it
is a genuinely novel figure for a merging paper.

### Honest expectations

* The baseline will probably beat the merged model on both perplexity and downstream
  accuracy. Plan for this. SAP-Pure is bounded above by the ensemble of its shards.
* The merged model should clearly beat **any individual shard** — that is the claim that
  merging recovered something.
* The merged model should *dramatically* beat the **weight-averaging baseline**. That
  contrast is the central hypothesis of report §12 and the most robust result you have.
* The healing pass (report §6.5b) is what closes the remaining gap. Report SAP-Pure and
  SAP-Heal as separate rows; the delta between them *is* the size of the composition gap.

---

## 9. Suggested order of work

1. `combine_partitions.py`, then `train_baseline.py --dry-run` → confirm `exact match`.
2. Train the baseline (the long pole — days).
3. While it trains: `generate.py` on the merged model and a shard; get a feel for output.
4. Baseline done → `benchmark.py` with `--models`/`--bins` for the perplexity table.
5. Fine-tune merged + baseline + one shard on AG News, 3 seeds, full fine-tune.
6. Repeat with `--freeze-backbone`.
7. `benchmark.py --finetune-root ... --compare merged baseline` for the full report.
8. If time allows: SST-2, the sample-efficiency curve, SAP-Heal.

---

## 10. Troubleshooting

| symptom | fix |
|---|---|
| `ModuleNotFoundError: datasets` | `pip install datasets` (or use `file:` datasets) |
| `dataset has at least K distinct labels but num_labels=N` | pass `--num-labels K` |
| accuracy stuck at chance | `--head-lr` too low, or `--tokenizer` does not match pretraining |
| accuracy 100% immediately | dataset is trivially separable — check your labels |
| loss NaN | lower `--lr`; try `--dtype fp32` |
| CUDA OOM | halve `--batch-size`, double `--grad-accum`; lower `--max-length` |
| `resume mismatch on 'seed'` | `--seeds` changed; use `--fresh` or restore |
| cache seems stale | delete `data/finetune_cache/` |
| generation is very slow | `--device cuda`; no KV cache means O(n²) |
| `--compare` finds no results | the `--name` values must match those used at fine-tune time |
