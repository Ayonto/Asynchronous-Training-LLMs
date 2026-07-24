#!/usr/bin/env python
"""Consolidated head-to-head report: merged model vs conventional baseline.

Pulls together every instrument the pipeline produces and prints one comparison:

  1. Language-modelling  — held-out perplexity on val.bin (and per-partition
                           val sets, which show whether the merged model kept
                           each shard's specialty).
  2. Downstream          — fine-tuned classification accuracy / macro-F1,
                           aggregated over seeds with a 95% confidence interval
                           on the DIFFERENCE, because that is the quantity the
                           thesis actually claims something about.
  3. Accounting          — parameters and pretraining tokens per model, so no
                           comparison is silently unfair.

Both sections are optional: run it with only --bins to get perplexity before
any fine-tuning exists, or only --finetune-root to get downstream results alone.

Examples
--------
# perplexity only (available as soon as the baseline finishes pretraining)
python scripts/benchmark.py \
    --models runs/merged/merged.pt runs/baseline_546m/final.pt \
    --bins data/fineweb_val/val.bin \
    --out runs/benchmark

# the full report
python scripts/benchmark.py \
    --models runs/merged/merged.pt runs/baseline_546m/final.pt \
             runs/shard_01/final.pt \
    --bins data/fineweb_val/val.bin data/fineweb_val/part_01.val.bin \
           data/fineweb_val/part_02.val.bin \
    --finetune-root runs/finetune \
    --compare merged baseline \
    --out runs/benchmark
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from sap.evaluate import evaluate_models, format_results_table  # noqa: E402


# 95% two-sided critical values for Student's t, indexed by degrees of freedom.
# A small table keeps this script dependency-free (no scipy); df beyond the
# table is clamped to the normal-approximation value.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086,
        25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t_critical(df: float) -> float:
    if df <= 0:
        return float("nan")
    keys = sorted(_T95)
    for k in keys:
        if df <= k:
            return _T95[k]
    return 1.960


def welch_ci(a: List[float], b: List[float]) -> Optional[dict]:
    """95% CI for mean(a) - mean(b) under unequal variances (Welch).

    Reported instead of a bare p-value because the effect SIZE and its
    uncertainty are what a reader needs; an interval that straddles zero says
    'not resolved at this seed count' far more clearly than 'p = 0.21'.
    """
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return {"diff": ma - mb, "se": 0.0, "lo": ma - mb, "hi": ma - mb,
                "df": float("inf"), "resolved": ma != mb}
    df_num = (va / na + vb / nb) ** 2
    df_den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = df_num / df_den if df_den > 0 else float("inf")
    t = t_critical(df)
    diff = ma - mb
    return {"diff": diff, "se": se, "lo": diff - t * se, "hi": diff + t * se,
            "df": df, "t_stat": diff / se,
            "resolved": (diff - t * se) * (diff + t * se) > 0}


def collect_finetune(root: Path) -> Dict[str, List[dict]]:
    """Gather every <name>_<dataset>_aggregate.json under the fine-tune root."""
    out: Dict[str, List[dict]] = {}
    if not root.exists():
        return out
    for path in sorted(root.glob("*_aggregate.json")):
        with open(path, "r", encoding="utf-8") as f:
            agg = json.load(f)
        key = f"{agg['dataset']}|{agg.get('mode', 'full_finetune')}"
        out.setdefault(key, []).append(agg)
    # fall back to per-run summaries if no aggregates were written
    if not out:
        by: Dict[str, Dict[str, List[dict]]] = {}
        for path in sorted(root.glob("*/summary.json")):
            with open(path, "r", encoding="utf-8") as f:
                s = json.load(f)
            key = f"{s['dataset']}|{s.get('mode', 'full_finetune')}"
            name = s["name"].rsplit("_s", 1)[0]
            by.setdefault(key, {}).setdefault(name, []).append(s)
        for key, models in by.items():
            for name, runs in models.items():
                accs = [r["best"]["accuracy"] for r in runs]
                f1s = [r["best"]["macro_f1"] for r in runs]
                out.setdefault(key, []).append({
                    "name": name,
                    "backbone_source": runs[0]["backbone_source"],
                    "dataset": runs[0]["dataset"],
                    "mode": runs[0].get("mode", "full_finetune"),
                    "seeds": [r["seed"] for r in runs],
                    "accuracy_mean": statistics.mean(accs),
                    "accuracy_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
                    "macro_f1_mean": statistics.mean(f1s),
                    "macro_f1_std": statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
                    "per_seed": [{"seed": r["seed"],
                                  "accuracy": r["best"]["accuracy"],
                                  "macro_f1": r["best"]["macro_f1"]} for r in runs],
                })
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--models", nargs="*", default=[],
                   help="checkpoints to evaluate for perplexity")
    p.add_argument("--bins", nargs="*", default=[],
                   help="held-out .bin files (val.bin, part_XX.val.bin, ...)")
    p.add_argument("--finetune-root", default=None,
                   help="directory containing the fine-tune aggregates")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None,
                   help="two --name labels to contrast, e.g. --compare merged baseline")
    p.add_argument("--out", default="runs/benchmark", help="report output directory")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=None,
                   help="cap eval batches per file (leave unset for the full file)")
    p.add_argument("--device", default="auto")
    p.add_argument("--fp32", action="store_true",
                   help="evaluate in fp32 for bit-stable numbers")
    p.add_argument("--data-dtype", default=None, choices=["uint16", "uint32"])
    args = p.parse_args()

    if not args.models and not args.finetune_root:
        p.error("give --models/--bins, or --finetune-root, or both")

    device = (("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    # =====================================================================
    # 1. Language modelling
    # =====================================================================
    if args.models and args.bins:
        print("\n" + "=" * 78)
        print("1. LANGUAGE MODELLING — held-out cross-entropy / perplexity")
        print("=" * 78)
        for m in args.models:
            if not Path(m).exists():
                p.error(f"checkpoint not found: {m}")
        for b in args.bins:
            if not Path(b).exists():
                p.error(f"token file not found: {b}")
        results = evaluate_models(
            args.models, args.bins, seq_len=args.seq_len,
            batch_size=args.batch_size, max_batches=args.max_batches,
            device=device, fp32=args.fp32, dtype=args.data_dtype,
        )
        print(format_results_table(results))
        report["language_modelling"] = results
        with open(out_dir / "perplexity.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("\n  Read the per-partition columns as a specialty-retention check: a")
        print("  merged model should be competitive on EVERY part_XX.val.bin, while")
        print("  each individual shard is strong only on its own.")
    elif args.models or args.bins:
        print("NOTE: perplexity needs both --models and --bins; skipping section 1.")

    # =====================================================================
    # 2. Downstream classification
    # =====================================================================
    if args.finetune_root:
        root = Path(args.finetune_root)
        groups = collect_finetune(root)
        if not groups:
            print(f"\nNOTE: no fine-tune results found under {root}; skipping section 2.")
        else:
            print("\n" + "=" * 78)
            print("2. DOWNSTREAM — fine-tuned classification (best checkpoint per run)")
            print("=" * 78)
            report["downstream"] = {}
            for key in sorted(groups):
                dataset, mode = key.split("|")
                rows = sorted(groups[key], key=lambda r: -r["accuracy_mean"])
                print(f"\n  dataset: {dataset}    mode: {mode}")
                print(f"  {'model':<22} {'seeds':>6} {'accuracy':>18} "
                      f"{'macro-F1':>18}")
                print("  " + "-" * 66)
                for r in rows:
                    ns = len(r.get("seeds", []) or r.get("per_seed", []))
                    print(f"  {r['name']:<22} {ns:>6} "
                          f"{r['accuracy_mean']:>10.4f} +- {r['accuracy_std']:<5.4f} "
                          f"{r['macro_f1_mean']:>10.4f} +- {r['macro_f1_std']:<5.4f}")
                report["downstream"][key] = rows

                if args.compare:
                    a_name, b_name = args.compare
                    by_name = {r["name"]: r for r in rows}
                    if a_name in by_name and b_name in by_name:
                        A = [x["accuracy"] for x in by_name[a_name]["per_seed"]]
                        B = [x["accuracy"] for x in by_name[b_name]["per_seed"]]
                        ci = welch_ci(A, B)
                        print(f"\n    {a_name} - {b_name} accuracy difference:")
                        if ci is None:
                            print("      need >= 2 seeds per model for an interval")
                        else:
                            verdict = ("RESOLVED — the interval excludes zero"
                                       if ci["resolved"] else
                                       "NOT RESOLVED — the interval includes zero")
                            print(f"      {ci['diff']:+.4f}  "
                                  f"95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]  "
                                  f"(Welch, df={ci['df']:.1f})")
                            print(f"      {verdict}")
                            if not ci["resolved"]:
                                print("      -> add seeds before claiming a difference.")
                            report["downstream"][key + "|comparison"] = ci
                    else:
                        missing = [n for n in (a_name, b_name) if n not in by_name]
                        print(f"\n    (--compare: no results named {missing} "
                              f"for this dataset/mode)")

    # =====================================================================
    # 3. Accounting
    # =====================================================================
    if "language_modelling" in report:
        print("\n" + "=" * 78)
        print("3. ACCOUNTING — is the comparison fair?")
        print("=" * 78)
        print(f"  {'model':<28} {'params':>10} {'pretrain tokens':>18}")
        print("  " + "-" * 58)
        acct = []
        for mp, row in report["language_modelling"].items():
            m = row["meta"]
            tok = m.get("tokens_seen")
            acct.append({"path": mp, "name": m["name"],
                         "params": m["params_total"], "tokens_seen": tok})
            print(f"  {(m['name'] or Path(mp).stem)[:28]:<28} "
                  f"{m['params_total'] / 1e6:>9.1f}M "
                  f"{(f'{tok:,}' if tok else '-'):>18}")
        report["accounting"] = acct
        print("\n  A clean comparison holds BOTH columns close. If the merged model")
        print("  and the baseline differ in parameters, say so explicitly; if they")
        print("  differ in tokens, the result is about compute, not about merging.")

    # =====================================================================
    with open(out_dir / "benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nfull report -> {out_dir / 'benchmark_report.json'}\n")


if __name__ == "__main__":
    main()
