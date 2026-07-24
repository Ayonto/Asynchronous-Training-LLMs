#!/usr/bin/env python
"""Sizing calculator: pick shard widths for a target merged model size.

Pure arithmetic (no torch) using the exact SAP parameter formulas:

  E                = vocab_size x d_model                     (tied embedding, shared)
  attn  per layer  = 4 x d_model x n_heads x d_head
  ffn   per layer  = 3 x d_model x d_ff                        (SwiGLU)
  shard total      = E + L x (attn + ffn + 2 x d_model) + d_model
  merged (N equal) = E + N x L x (attn + ffn) + (2L + 1) x d_model

Examples
--------
# what do I get from 5 shards of H=8, d_ff=1024 on the reference family?
python scripts/plan_model.py --family configs/family_reference.json \
    --n-heads 8 --d-ff 1024 --num-shards 5

# solve d_ff for a ~1B merged model with 5 shards at H=8:
python scripts/plan_model.py --family configs/family_reference.json \
    --n-heads 8 --num-shards 5 --target-merged 1000000000
"""

import argparse
import json
import sys
from pathlib import Path


def plan(fam: dict, n_heads: int, d_ff: int, n_shards: int) -> dict:
    L, dm, dh, V = fam["n_layers"], fam["d_model"], fam["d_head"], fam["vocab_size"]
    E = V * dm
    attn = 4 * dm * n_heads * dh
    ffn = 3 * dm * d_ff
    per_layer = attn + ffn + 2 * dm
    shard = E + L * per_layer + dm
    merged = E + n_shards * L * (attn + ffn) + (2 * L + 1) * dm
    return {"E": E, "attn_per_layer": attn, "ffn_per_layer": ffn,
            "shard_stackable": shard - E, "shard_total": shard, "merged": merged}


def solve_dff(fam: dict, n_heads: int, n_shards: int, target_merged: float) -> int:
    """Smallest d_ff (rounded up to a multiple of 64) whose merged size >= target."""
    L, dm, dh = fam["n_layers"], fam["d_model"], fam["d_head"]
    E = fam["vocab_size"] * dm
    attn = 4 * dm * n_heads * dh
    budget_per_layer = (target_merged - E - (2 * L + 1) * dm) / (n_shards * L)
    ffn_budget = budget_per_layer - attn
    if ffn_budget <= 0:
        sys.exit(f"ERROR: n_heads={n_heads} alone already exceeds the target; "
                 "lower --n-heads or raise --target-merged")
    dff = ffn_budget / (3 * dm)
    return max(64, int((dff + 63) // 64) * 64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, help="family JSON path")
    ap.add_argument("--n-heads", type=int, required=True)
    ap.add_argument("--d-ff", type=int, default=None)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--target-merged", type=float, default=None,
                    help="solve for d_ff so the N-way merge lands at/above this size")
    args = ap.parse_args()

    fam = json.loads(Path(args.family).read_text())
    if args.d_ff is None:
        if args.target_merged is None:
            ap.error("give --d-ff, or --target-merged to solve for it")
        args.d_ff = solve_dff(fam, args.n_heads, args.num_shards, args.target_merged)
        print(f"solved: d_ff = {args.d_ff} (rounded up to a multiple of 64)\n")

    p = plan(fam, args.n_heads, args.d_ff, args.num_shards)
    L, dm = fam["n_layers"], fam["d_model"]
    print(f"family: L={L}  d_model={dm}  d_head={fam['d_head']}  "
          f"vocab={fam['vocab_size']}  |  shard: H={args.n_heads}  d_ff={args.d_ff}  "
          f"|  N={args.num_shards}")
    print("-" * 74)
    print(f"embedding E (shared, counted once)   : {p['E'] / 1e6:>9.1f}M")
    print(f"per-layer attention / ffn            : {p['attn_per_layer'] / 1e6:>9.2f}M / "
          f"{p['ffn_per_layer'] / 1e6:.2f}M")
    print(f"ONE SHARD: stackable + E = total     : {p['shard_stackable'] / 1e6:>9.1f}M + "
          f"{p['E'] / 1e6:.1f}M = {p['shard_total'] / 1e6:.1f}M")
    print(f"MERGED ({args.num_shards}-way)                       : "
          f"{p['merged'] / 1e6:>9.1f}M  "
          f"(heads {args.num_shards * args.n_heads}/layer, "
          f"d_ff {args.num_shards * args.d_ff})")
    print("-" * 74)
    chin = 20 * p["shard_total"]
    opt_bytes = 16 * p["shard_total"]           # fp32 weights+grads+AdamW moments
    print(f"training-side rough numbers per shard:")
    print(f"  Chinchilla-optimal tokens (~20/param): {chin / 1e9:>6.1f}B "
          f"(=> each partition should hold at least this many tokens)")
    print(f"  static GPU memory (weights+grads+Adam, fp32): {opt_bytes / 1e9:>5.1f} GB "
          f"(+ activations/logits: depends on batch_size x seq_len — start small, "
          f"watch nvidia-smi, raise)")
    print(f"  checkpoint file size (with optimizer): ~{12 * p['shard_total'] / 1e9:.1f} GB; "
          f"disk per shard with keep-prev 3 is ~{4 * 12 * p['shard_total'] / 1e9:.0f} GB")


if __name__ == "__main__":
    main()
