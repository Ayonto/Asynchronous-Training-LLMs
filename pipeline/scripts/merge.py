#!/usr/bin/env python
"""Merge N trained family members into one model (pure tensor surgery).

Every stack merge is gated by an exactness check: the merged model must
reproduce the claimed per-sublayer identity against its parents (block by
block, float64) or the script refuses to save.

Examples
--------
# token-weighted (alpha_k = tokens_k / total), scaled — the thesis default:
python scripts/merge.py --inputs runs/shard_01/final.pt runs/shard_02/final.pt \
    runs/shard_03/final.pt --out runs/merged_123.pt

# unscaled variant (function SUM instead of average):
python scripts/merge.py --inputs runs/shard_01/final.pt runs/shard_02/final.pt \
    --out runs/merged_12_unscaled.pt --no-scale

# manual alphas / uniform alphas:
python scripts/merge.py --inputs A.pt B.pt --out M.pt --alpha-mode manual --alphas 0.7 0.3
python scripts/merge.py --inputs A.pt B.pt --out M.pt --alpha-mode uniform

# continual merge (a merged model is a family member; token bookkeeping is automatic):
python scripts/merge.py --inputs runs/merged_123.pt runs/shard_04/final.pt --out runs/merged_1234.pt

# naive weight-averaging baseline (identical widths only):
python scripts/merge.py --inputs A.pt B.pt --out avg_AB.pt --method avg
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sap.merge import merge_checkpoints  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True, help="2+ checkpoint paths")
    p.add_argument("--out", required=True, help="output checkpoint path")
    p.add_argument("--alpha-mode", choices=["tokens", "uniform", "manual"],
                   default="tokens")
    p.add_argument("--alphas", nargs="+", type=float, default=None,
                   help="required with --alpha-mode manual; normalized to sum to 1")
    p.add_argument("--no-scale", action="store_true",
                   help="skip the alpha scaling of wo / w_down: merged sublayers "
                        "compute the SUM of parent sublayers instead of the average "
                        "(embeddings are still averaged — they cannot be summed)")
    p.add_argument("--method", choices=["stack", "avg"], default="stack",
                   help="stack = SAP structural merge; avg = naive weight-average baseline")
    p.add_argument("--skip-check", action="store_true",
                   help="skip the exactness gate (not recommended)")
    p.add_argument("--check-tol", type=float, default=1e-3)
    p.add_argument("--name", default=None)
    args = p.parse_args()

    if len(args.inputs) < 2:
        p.error("need at least two checkpoints to merge")
    if args.alpha_mode == "manual" and args.alphas is None:
        p.error("--alpha-mode manual requires --alphas")

    meta, report = merge_checkpoints(
        checkpoint_paths=args.inputs,
        out_path=args.out,
        alpha_mode=args.alpha_mode,
        manual_alphas=args.alphas,
        scaled=not args.no_scale,
        method=args.method,
        check=not args.skip_check,
        tol=args.check_tol,
        name=args.name,
    )

    print("\n=== merge complete ===")
    print(f"method: {meta['method']}   scaled: {meta['scaled']}   "
          f"alpha mode: {meta['alpha_mode']}")
    for entry in meta["lineage"]:
        tok = entry["tokens_seen"]
        print(f"  {entry['name']:<24} alpha={entry['alpha']:.4f}  "
              f"tokens={tok:,}" if tok else
              f"  {entry['name']:<24} alpha={entry['alpha']:.4f}")
    print(f"total tokens behind merged model: {meta['tokens_seen']:,}")
    if report is not None:
        print(f"exactness gate PASSED: max error {report['max_err']:.3e} "
              f"(tol {report['tol']:.1e})")
    elif meta["method"] == "stack":
        print("WARNING: exactness gate was skipped (--skip-check)")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
