# SAP codebase — function-by-function explanation

This document explains every file and every function: what it does, why it exists, and
where the thesis math lives in it. Read it top to bottom once; afterwards use it as a
reference. (Plain-language companion to the code; the code comments state the same
contracts more tersely.)

Convention used throughout: PyTorch's `nn.Linear` stores its weight as
`(out_features, in_features)`. So "concatenate heads on the input side" means
`torch.cat(..., dim=0)` for wq/wk/wv (their *output* dimension is the head axis) and
`torch.cat(..., dim=1)` for wo (its *input* dimension is the head axis). Getting these
two dims right **is** the merge; everything else is bookkeeping.

---

## sap/config.py — the family contract

### `FamilyConfig` (dataclass)
Holds the skeleton every mergeable model must share: `n_layers`, `d_model`, `d_head`,
`vocab_size`, `max_seq_len`, `rope_theta`, `norm_eps`.

- `__post_init__` validates that `d_model % d_head == 0`. This does **not** force models
  to have `d_model/d_head` heads — head count is free — it only guarantees a *balanced*
  configuration exists in the family (a natural seed shape).
- `to_dict` / `from_dict` / `from_json` / `save_json` — serialization. The family dict is
  embedded into every checkpoint, so a checkpoint is self-describing.
- `__eq__` compares field-by-field; used to refuse merging models from different families.

### `ModelWidths` (dataclass)
Per-layer lists `n_heads[i]`, `d_ff[i]`. Lists (not scalars) because merged models have
summed widths and nothing forces them uniform.

- `uniform(n_layers, n_heads, d_ff)` — the common case: same widths at every layer.
- `from_spec(n_layers, "4", "1024")` — parses CLI strings; `"4"` broadcasts to all layers,
  `"4,4,8,8"` sets each layer individually (must match `n_layers`).

### `merged_widths(widths_list)`
The width arithmetic of a merge: per layer, `n_heads` and `d_ff` of the merged model are
the sums over parents. Raises if the parents disagree on layer count (depth cannot merge —
composition, not summation).

---

## sap/model.py — the transformer and checkpoint I/O

Built from scratch (no prebuilt transformer modules) so the merge can do surgery on
matrices whose exact layout we control.

### `RMSNorm`
`y = weight * x / sqrt(mean(x²) + eps)`. Chosen over LayerNorm deliberately: the gain
`weight` is a *diagonal* multiplication after a *parameter-free* normalization, which is
what makes exact gain absorption possible in the merge (`W @ (g*u) == (W*g) @ u`).
LayerNorm's bias would break that identity.

### `build_rope_cache(max_seq_len, d_head, theta)`
Precomputes RoPE angle tables `cos, sin` of shape `(max_seq_len, d_head/2)` in
**float64** (cast down at use). Frequencies `theta^(-2i/d_head)`. Since `theta` and
`d_head` are skeleton-fixed, every family member computes literally identical tables —
one reason attention merges exactly: position handling is per-head and parameter-free.

### `apply_rope(x, cos, sin)`
Rotates q or k, Llama-style half-split convention: split the head dim in two halves
`(x1, x2)` and output `[x1·cos − x2·sin, x1·sin + x2·cos]`. Any consistent convention
works; consistency across the family is what matters.

### `Attention`
Standard causal MHA with the crucial property that each head is fully self-contained:

- `wq/wk/wv: Linear(d_model → n_heads*d_head)` — head h owns rows `[h·d_head, (h+1)·d_head)`.
- `wo: Linear(n_heads*d_head → d_model)` — head h owns the matching *columns* of the
  weight (row-block of the math-notation W_O). This is the per-head "translator" into
  residual coordinates.
- `forward`: project → reshape to `(B, H, T, d_head)` → RoPE on q,k →
  `F.scaled_dot_product_attention(is_causal=True)` (uses the fused kernel on GPU, exact
  math fallback in float64 on CPU — which the exactness gate relies on) → reshape → `wo`.

`n_heads` is an attribute, but loading always *derives* it from weight shapes
(`widths_from_state_dict`) — nothing ever assumes `H·d_head == d_model`, so over-complete
merged models (rectangular wo) work everywhere.

### `SwiGLU`
`FFN(x) = w_down(SiLU(w_gate x) * (w_up x))`. Neuron j owns row j of `w_gate`, row j of
`w_up`, column j of `w_down`. The layer output is the sum of per-neuron contributions —
the additivity the FFN merge rests on.

### `Block`
Pre-norm residual block: `x += attn(attn_norm(x)); x += ffn(ffn_norm(x))`. "Sublayer" in
the thesis means `attn ∘ attn_norm` or `ffn ∘ ffn_norm` — the exactness identity is
stated and verified at exactly this granularity.

### `SAPModel`
`embed → blocks → final_norm → tied LM head` (logits are `x @ embed.weightᵀ` via
`F.linear`; tying is a skeleton decision — one embedding matrix to average per merge, not
two).

- `__init__` builds blocks from the per-layer widths, registers the RoPE cache as
  non-persistent buffers (not saved in checkpoints; rebuilt from the family config), and
  applies GPT-2-style init: N(0, 0.02) everywhere, residual-output projections
  (`wo`, `w_down`) scaled by `1/sqrt(2·n_layers)`.
- `forward(idx, targets)` returns `(logits, loss)`; loss is cross-entropy computed in
  float32 even under bf16 autocast (numerical hygiene).
- `generate` — plain sampling loop (temperature, top-k) for qualitative checks.
- `param_counts()` returns `{total, embedding, stackable}` — the E vs. S−E split of the
  growth law `S_merged = E + Σ(Sᵢ−E)`.

### Checkpoint I/O

- `widths_from_state_dict(state, family)` — reads per-layer head counts (`wq` rows ÷
  `d_head`) and FFN widths (`w_gate` rows) from the actual tensors. The report's cardinal
  rule ("derive H from weight shapes, never from a constant") implemented once, used at
  every load.
- `atomic_torch_save(payload, path)` — writes to a temp file in the same directory, then
  `os.replace`. Rename is atomic, so a crash mid-save leaves the previous checkpoint
  intact. This plus resume is the crash-safety story.
- `save_model_checkpoint(path, model, meta, optimizer_state, train_state, extra)` — the
  one canonical schema:

  ```
  format_version, family (dict), widths (dict), model_state,
  meta {role, name, tokens_seen, lineage?, ...},
  optimizer_state?, train_state?, train_args?
  ```

  `meta.tokens_seen` is load-bearing: it is what token-weighted merge alphas are computed
  from, including through chains of continual merges.
- `load_checkpoint` / `model_from_checkpoint` / `load_model` — load, verify that stored
  widths match the widths derived from the weights (corruption/tampering check), rebuild.

---

## sap/data.py — corpus preparation and deterministic sampling

### `dtype_for_vocab(vocab_size)`
`uint16` when the vocab fits (≤ 65536), else `uint32`. Halves disk/RAM for typical vocabs.

### `find_meta(bin_path)` / `load_tokens(bin_path, dtype)`
Every prepared data directory carries a `meta.json`; `load_tokens` memory-maps a `.bin`
using the dtype recorded there (or an explicit override). Memmap means a 50GB token file
costs no RAM — the sampler reads only the chunks it needs.

### `BinWriter`
Buffered appender: accumulates documents' token arrays and flushes to disk every ~1M
tokens. Tracks token and document counts for `meta.json`.

### `iter_documents(inputs, text_key, txt_mode)`
Uniform document stream over heterogeneous sources, yielding
`(doc_index, input_index, text)`:
`.jsonl` (one object per line), `.txt` (per line or whole-file), `hf:` specs (streamed via
`datasets`, so corpora larger than disk/RAM work). `doc_index` is global and sequential —
it is the routing key.

### `_hash_unit(tag, seed, i)`
Deterministic uniform [0,1) from `crc32(f"{tag}:{seed}:{i}")`. Used instead of Python's
`hash()`/RNG streams because crc32 is stable across platforms, Python versions, and
`PYTHONHASHSEED` — the guarantee that partitions are reproducible anywhere.

### `route_document(...)`
The partitioning policy, applied per document, in order:
1. global val with probability `val_fraction` (checked FIRST — so validation text can
   never leak into any training file, including the seed sample);
2. otherwise a partition, by mode: `random` (hash → i.i.d. partitions), `blocks`
   (contiguous runs of `block_docs` documents round-robined), `by-input` (input file k →
   partition k; true domain split).

### `prepare_dataset(...)`
Orchestrates: tokenizer loaded (HF `AutoTokenizer` — lazy import, only data prep needs
transformers) → documents streamed → batched tokenization (`encode_batch_docs` at a time,
significant speedup) → routing → EOS appended to every document → writers → `meta.json`.
Two extra routes on training documents: per-partition validation (`pval` hash <
`partition_val_fraction` → `part_k.val.bin`) and the seed sample (`seed` hash <
`seed_fraction` → document *also* copied to `seed.bin`; the overlap with partitions is
intentional — the seed is supposed to see a mixed sample of D). Errors out if the
tokenizer has no EOS and none is supplied.

### `ChunkSampler`
The deterministic-resume engine. The token stream is cut into non-overlapping chunks of
`seq_len` (+1 lookahead for targets); an epoch is a permutation of all chunk indices
seeded by `data_seed + epoch`; batches are consecutive slices of the permutation.

- `next_batch()` returns the next batch's chunk indices and advances `(epoch, cursor)`.
  Epoch rollover is *eager* (happens when the last batch of an epoch is returned), so
  `sampler.epoch` always equals "epochs fully completed" — which is what the
  `--max-epochs` budget checks.
- `state()` → `{epoch, cursor}` — two integers fully describe the data position, because
  the permutation itself is recomputed from the seed. This is why checkpoints are small
  and resume is exact.
- Partial trailing batches are dropped (standard practice; keeps every step the same size
  so token accounting is exact).

### `get_batch(tokens, chunk_idxs, seq_len, device)`
Materializes `(x, y)` int64 tensors; `y` is `x` shifted one position (next-token
prediction). Pinned-memory + non-blocking transfer on CUDA.

---

## sap/merge.py — the thesis operator

### `absorb_norm_gains(model)` (in place; exported mainly for tests)
For each block: multiply `wq/wk/wv` input columns by the attention-norm gain, `w_gate/w_up`
by the FFN-norm gain, set the gains to 1. Exact because RMSNorm's gain is diagonal and
rms() is parameter-free. After absorption every head/neuron sees exactly the normalized
input it was trained on, so merging never needs to average norm gains. The final
pre-LM-head norm is the exception (its consumer, the LM head, is shared and averaged), so
it stays and is α-averaged in the merge — a known, small approximation stated in the thesis.

### `resolve_alphas(n_models, alpha_mode, manual_alphas, token_counts)`
Produces the convex weights:
`tokens` → αₖ = tokensₖ/Σtokens (thesis default; errors if a model has no recorded
tokens — no silent fallback), `uniform` → 1/N, `manual` → user values normalized to sum
to 1. Convexity matters: weights summing to 1 keep merged sublayer outputs in the
magnitude regime every downstream layer was trained for.

### `merge_models(models, alphas, scaled=True)`
The operator itself. Never mutates the parents (absorption is computed on the fly into
fresh tensors). All arithmetic in float64, cast to the parents' dtype once at the end —
so float64 parents merge *exactly* (how the unit tests pin ~1e-13) and float32 parents
lose only the unavoidable single storage rounding.

Per block, with `s_k = alpha_k` if `scaled` else `1`:

```
wq_m = cat_k( wq_k * g_attn_k , dim=0)     # heads side by side, gain absorbed, UNSCALED
wk_m, wv_m                                  same
wo_m = cat_k( s_k * wo_k      , dim=1)     # head translators, SCALED
w_gate_m = cat_k( w_gate_k * g_ffn_k, dim=0)   # neurons, absorbed, UNSCALED
w_up_m   =                          same
w_down_m = cat_k( s_k * w_down_k, dim=1)       # neuron outputs, SCALED
norm gains -> 1
```

Embedding and final norm: `Σ αₖ ·(·)` — always the convex alphas, even in unscaled mode
(summing embeddings would inflate the input stream N-fold; there is no meaningful
"unscaled" version of a shared input table).

Why input side unscaled / output side scaled: scaling wq/wk/wv would change what each
head computes internally (queries, keys, softmax); scaling wo scales the head's finished
contribution, which combines linearly. Same logic for w_gate/w_up vs. w_down.

### `weight_average(models, alphas)`
The baseline: plain α-weighted average of every parameter. Requires identical widths
(otherwise shapes don't even align — itself a talking point). This is the method the
thesis predicts degrades catastrophically as shards diverge (permutation symmetry).

### `verify_merge(parents, merged, alphas, scaled, ...)` — the exactness gate
For every layer: deep-copy the merged block and all parent blocks to CPU float64, feed
the same random input, and check

- scaled: `merged_sublayer(x) == Σ αₖ · parent_k_sublayer(x)`
- unscaled: `merged_sublayer(x) == Σ parent_k_sublayer(x)`

against the **original, un-absorbed** parents — so the gate covers the absorption step
too, not just concatenation. Also checks the embedding/final-norm averages and that
merged widths equal the per-layer sums. Block-at-a-time deep copies keep memory trivial
even for large models; tiny inputs (8 tokens) keep it fast.

Tolerance semantics: float64 models → ~1e-13 observed. Float32 models → ~1e-8..1e-5
observed (pure float32 storage rounding of the absorb/scale arithmetic; the identity
itself is proven separately in float64 by the tests). Default `tol=1e-3` sits far above
rounding and far below any real bug — a wrong dim/scale/slice yields errors of order 1
(the negative test proves the gate catches this). Raises `MergeExactnessError` → the
merge script never saves a bad model.

### `merge_checkpoints(paths, out, ...)`
The file-level wrapper the CLI calls: load checkpoints → pull `tokens_seen` from each
`meta` → `resolve_alphas` → `merge_models` (or `weight_average`) → `verify_merge` (stack
only; no identity exists for averaging) → write the merged checkpoint with full metadata:
method, scaled flag, α-mode, `tokens_seen = Σ parents` (this makes continual merging's
bookkeeping automatic — the report's rule "weight a merged model by its accumulated
tokens" falls out of the metadata), and a `lineage` list recording every parent's name,
role, tokens, and α.

---

## sap/train.py — the crash-safe trainer

### `TrainConfig` (dataclass)
Every knob, grouped: identity/io, family+widths, initialization, optimization, budget
(`max_epochs` / `max_hours` / `max_steps`), checkpoint cadence
(`checkpoint_every_min` / `checkpoint_every_steps`, `keep_history`), logging/eval,
hardware (`device`, `dtype`, `compile`). One config = one model's training run; per-model
control of cadence and budget is just "each shard has its own config".

### `RESUME_CRITICAL_FIELDS`
`data_path, batch_size, grad_accum, seq_len, data_seed, n_heads_spec, d_ff_spec,
init_from`. Changing any of these between crash and resume would silently invalidate the
sampler position or the model identity, so resume compares them against the values stored
in the checkpoint (`train_args`) and refuses with an explanation. `--fresh` is the
deliberate escape hatch. Budgets and LR settings are *not* critical — extending
`--max-hours` on resume is legitimate.

### `grow_from_checkpoint(init_path, family, target_widths, init_seed)`
Branching logic:
- `target_widths=None` or equal → exact copy of the seed.
- wider → **function-preserving growth**: seed heads/neurons are copied into the first
  slots; new units get random *input-side* weights (they need diversity to learn distinct
  features) but **zero output-side weights** (`wo` / `w_down` columns), so the grown model
  computes exactly the seed's function at branch time while new units still receive
  gradient (their inputs produce activations; the zero output weights get nonzero grads)
  and differentiate during shard training. Verified by
  `test_grow_from_seed_is_function_preserving`.
- narrower → refused (shrinking is not defined).
Also refuses a family mismatch. Returns the seed's meta so its token count can be recorded
separately (`seed_tokens` — informational; a shard's α counts only its own partition
tokens, otherwise every shard would double-count the same seed tokens).

### `configure_optimizer(model, cfg)`
AdamW, betas (0.9, 0.95), weight decay only on matrices (dim ≥ 2) — norm gains undecayed,
standard LLM practice. `fused=True` on CUDA.

### `lr_at(step, cfg, total_steps)`
Functional LR: linear warmup → cosine to `min_lr` over `total_steps` (or constant).
A pure function of the step count means resume needs no scheduler state. `total_steps`
comes from `max_steps`, else `steps_per_epoch × max_epochs`, else the schedule falls back
to constant with a warning (a cosine needs a horizon). Consequence (documented in the
README): under cosine, extending the budget changes future LRs; use `constant` if you
plan to extend runs.

### `run_training(cfg)` — the loop, in order
1. **Data**: memmap the partition; cross-check `meta.json` vocab against the family
   (tokenizer-mismatch guard); optional val memmap.
2. **Model**: three-way init — *resume* (found `out_dir/latest.pt`, `--fresh` not set:
   rebuild model, restore optimizer/step/tokens/elapsed/sampler/RNG, enforce
   `RESUME_CRITICAL_FIELDS`) / *branch* (`--init-from`: `grow_from_checkpoint`) /
   *scratch* (`torch.manual_seed(init_seed)` then random init — give each independent
   shard its own seed). `torch.compile` wraps the forward only; saving always uses the
   raw module.
3. **Sampler**: `ChunkSampler` at the restored `(epoch, cursor)`.
4. **Budget checks** at the top of every step: `max_steps`, then `max_epochs` (sampler's
   eager epoch counter), then `max_hours` — against `elapsed_prev + this session`, so wall
   time accumulates correctly across any number of crashes/resumes.
5. **Step**: functional LR → `grad_accum` micro-batches (forward under bf16 autocast on
   CUDA, loss scaled by `1/grad_accum`, backward) → global-norm clip → optimizer step.
   `tokens_seen += batch·seq` per micro-batch — the number that becomes the shard's merge α.
6. **Logging**: running mean loss to stdout and `train_log.jsonl` (one JSON per line —
   trivially plottable); optional quick val eval every `eval_interval` steps.
7. **Checkpointing**: when the per-model time interval or step interval fires → atomic
   save of `latest.pt` (model + optimizer + train state + RNG states + args); optional
   rotating step-stamped snapshots (`keep_history`).
8. **Finish**: final `latest.pt` (resumable) + `final.pt` (no optimizer/train state —
   half the size; the file you carry between machines and feed to merge/eval). Returns a
   summary dict (also used by the tests).

Meta written into every save: role (seed/shard), name, `tokens_seen`, `seed_tokens`,
`init_from`, data path, steps, epochs completed, elapsed seconds.

---

## sap/evaluate.py — measurement

### `evaluate_bin(model, tokens, ...)`
Walks the file in *sequential, non-overlapping* windows (no sampling → reproducible
numbers), accumulating token-weighted cross-entropy and top-1 next-token accuracy.
Returns `{loss, perplexity, top1_accuracy, tokens_evaluated, batches}`. bf16 autocast on
CUDA unless `fp32=True`. `max_batches` caps cost for quick passes.

### `evaluate_models(model_paths, bin_paths, ...)`
The dynamic harness: loads each checkpoint in turn (freeing GPU memory between models),
evaluates on every file, and attaches provenance (role, name, `tokens_seen`, parameter
counts, widths) so the results JSON is self-describing.

### `format_results_table(results)` / `save_results(results, path)`
Console grid (models × files: loss/ppl/acc) and the JSON dump.

### `generate_text(model_path, tokenizer_name, prompt, ...)`
Loads the tokenizer (lazy transformers import) and samples — qualitative checks of
shards vs. merged models.

---

## scripts/ — thin CLIs

Each script only parses arguments and calls the library (three `sys.path` lines make the
repo importable without installation). All logic is in `sap/`, so notebooks or your own
experiments can import the same functions the CLIs use.

- **prepare_data.py** — args → `prepare_dataset`; prints per-file token/document counts
  and the vocab size to copy into the family JSON.
- **train.py** — args → `TrainConfig` → `run_training`. Requires at least one budget
  flag. Width flags are optional with `--init-from` (inherit) and required from scratch.
- **merge.py** — args → `merge_checkpoints`. Prints the α table, total token bookkeeping,
  and the gate's max error. `--no-scale`, `--alpha-mode`, `--method avg`, `--skip-check`
  (discouraged), `--check-tol`.
- **evaluate.py** — args → `evaluate_models` (+ optional `generate_text`); prints the
  grid, writes JSON.

---

## tests/test_exactness.py — what each test proves

| test | claim it pins down |
|---|---|
| `test_pairwise_heterogeneous_exactness` | 2+3-head attention and 6+10-neuron FFN with random norm gains: merged sublayer = 0.4·A + 0.6·B, float64 ~1e-13 |
| `test_nway_exactness` | 3 models, α=(0.5,0.3,0.2), one shot — N-way is native, not iterated pairwise |
| `test_sequential_equals_simultaneous` | merge(merge(A,B;½,½),C;⅔,⅓) equals merge(A,B,C;⅓ each) weight-for-weight — closure + α bookkeeping |
| `test_overcomplete_wo` | 6+8 heads at d_model=16/d_head=4: W_O rectangular (16×56), exactness unaffected — the "balanced config is a coincidence" insight |
| `test_unscaled_merge_is_exact_sum` | scaled=False gives exactly A+B (your no-scaling condition is a *different exact identity*, not a sloppy variant) |
| `test_norm_absorption_preserves_function` | absorption is a no-op on the model's function |
| `test_verify_merge_passes_and_reports` / `_catches_corruption` / `_unscaled_mode` | the gate passes correct merges, **fails** a corrupted slice, and distinguishes sum from average |
| `test_weight_average_requires_identical_widths` | baseline defined only for equal widths; correct averaging when defined |
| `test_alpha_resolution` | token-weighted / uniform / manual α; zero-token models rejected |
| `test_parameter_growth_accounting` | S_merged = E + Σ(Sᵢ−E) − (N−1)(2L+1)d_model, exact to the parameter |
| `test_checkpoint_roundtrip` / `test_widths_derived_from_weights` | save→load identity; widths always derivable from tensors |
| `test_grow_from_seed_is_function_preserving` | wider branch computes the seed's exact function at step 0 |
| `test_chunk_sampler_resume_and_coverage` | sampler resume is exact; one epoch covers each chunk once |
| `test_crash_resume_reproduces_uninterrupted_run` | 4 steps + crash + resume to 8 ≡ 8 straight steps, weight-for-weight |
| `test_resume_rejects_changed_critical_args` | silent-corruption guard works |
| `test_epoch_budget_stops_training` | `max_epochs` stops exactly at the boundary |
| `test_end_to_end_train_merge_verify` | full file-level pipeline: train two unequal shards → merge → token-α 1:2 → gate green → widths/meta correct |

---

## The checkpoint schema (reference)

```python
{
  "format_version": 1,
  "family":  {n_layers, d_model, d_head, vocab_size, max_seq_len, rope_theta, norm_eps},
  "widths":  {"n_heads": [...], "d_ff": [...]},         # per layer
  "model_state": <state_dict>,
  "meta": {
    "role": "seed" | "shard" | "merged" | "baseline_avg",
    "name": str,
    "tokens_seen": int,          # merge alphas are computed from this
    "seed_tokens": int,          # shards only; informational
    "lineage": [...],            # merged only: parents' names/roles/tokens/alphas/paths
    ...
  },
  # training checkpoints (latest.pt) additionally:
  "optimizer_state": ...,
  "train_state": {step, tokens_seen, epoch, cursor, elapsed_seconds, rng_*},
  "train_args": {...},           # for resume-consistency checking
}
```

`final.pt` = the same minus optimizer/train_state: the portable artifact you move between
machines and hand to `merge.py` / `evaluate.py`.
