#!/usr/bin/env python
"""Fine-tune any family checkpoint (merged, baseline, or shard) for classification.

Works identically on every checkpoint the pipeline produces, which is the point:
the merged model and the conventional baseline are fine-tuned by the same code,
on the same tokenized cache, with the same hyperparameters, so the only variable
is the backbone.

Run >= 3 seeds per model. Fine-tuning variance at this scale is comfortably
larger than the effects being measured, and a single-seed comparison is not
evidence. `--seeds 0 1 2` does this in one command and prints mean +- std.

Examples
--------
# list the built-in datasets and exit
python scripts/finetune.py --list-datasets

# fine-tune the merged model on AG News, 3 seeds
python scripts/finetune.py \
    --init-from runs/merged/merged.pt --name merged \
    --dataset ag_news --out-root runs/finetune \
    --seeds 0 1 2 --max-epochs 3 --batch-size 16 --max-length 256

# same for the baseline (identical flags -> a fair comparison)
python scripts/finetune.py \
    --init-from runs/baseline_546m/final.pt --name baseline \
    --dataset ag_news --out-root runs/finetune \
    --seeds 0 1 2 --max-epochs 3 --batch-size 16 --max-length 256

# linear probe: measures representation quality with no fine-tuning confound
python scripts/finetune.py --init-from runs/merged/merged.pt --name merged_probe \
    --dataset ag_news --out-root runs/finetune --seeds 0 1 2 \
    --freeze-backbone --head-lr 1e-3 --max-epochs 5

# a different dataset
python scripts/finetune.py --init-from runs/merged/merged.pt --name merged_sst2 \
    --dataset sst2 --out-root runs/finetune --seeds 0 1 2

# any HuggingFace classification dataset
python scripts/finetune.py --init-from runs/merged/merged.pt --name merged_custom \
    --dataset hf:yelp_polarity --num-labels 2 --text-key text --label-key label \
    --out-root runs/finetune --seeds 0

# local files
python scripts/finetune.py --init-from runs/merged/merged.pt --name merged_local \
    --dataset file:mydata/train.jsonl --eval-file mydata/test.jsonl \
    --num-labels 3 --text-key text --label-key label --out-root runs/finetune
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sap.finetune import (  # noqa: E402
    FinetuneConfig,
    dataset_tag,
    describe_datasets,
    run_finetune,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--list-datasets", action="store_true",
                   help="print the built-in dataset registry and exit")

    # identity / io
    p.add_argument("--init-from", help="backbone checkpoint to fine-tune")
    p.add_argument("--name", help="short label for this model, e.g. 'merged'")
    p.add_argument("--out-root", default="runs/finetune",
                   help="results go to <out-root>/<name>_<dataset>_seed<K>/")

    # data
    p.add_argument("--dataset", default="ag_news",
                   help="registry name, hf:NAME[:CONFIG], or file:PATH")
    p.add_argument("--num-labels", type=int, default=None)
    p.add_argument("--text-key", default=None)
    p.add_argument("--label-key", default=None)
    p.add_argument("--train-split", default=None)
    p.add_argument("--eval-split", default=None)
    p.add_argument("--train-file", default=None)
    p.add_argument("--eval-file", default=None)
    p.add_argument("--max-train", type=int, default=None,
                   help="subsample the train split (smoke tests / low-resource runs)")
    p.add_argument("--max-eval", type=int, default=None)
    p.add_argument("--max-length", type=int, default=256,
                   help="tokens per example; truncation keeps the TAIL")
    p.add_argument("--cache-dir", default="data/finetune_cache")
    p.add_argument("--tokenizer", default="gpt2",
                   help="MUST match the tokenizer the backbone was pretrained with")
    p.add_argument("--tokenizer-backend", choices=["tiktoken", "hf"], default="tiktoken")

    # optimization
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="one run per seed; >= 3 recommended")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5, help="backbone LR")
    p.add_argument("--head-lr", type=float, default=1e-3, help="classification-head LR")
    p.add_argument("--warmup-frac", type=float, default=0.06)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--freeze-backbone", action="store_true",
                   help="linear probe: train only the head")

    # budget
    p.add_argument("--max-epochs", type=int, default=3)
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None)

    # checkpointing
    p.add_argument("--checkpoint-every-min", type=float, default=20.0)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--no-save-best", action="store_true")

    # logging / eval
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--eval-per-epoch", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=32)

    # hardware
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp32"], default="auto")
    args = p.parse_args()

    if args.list_datasets:
        print(describe_datasets())
        return
    if not args.init_from or not args.name:
        p.error("--init-from and --name are required (or use --list-datasets)")
    if not Path(args.init_from).exists():
        p.error(f"checkpoint not found: {args.init_from}")

    ds_tag = dataset_tag(args.dataset)
    summaries = []
    for seed in args.seeds:
        out_dir = Path(args.out_root) / f"{args.name}_{ds_tag}_seed{seed}"
        print("\n" + "=" * 70)
        print(f"FINE-TUNE  model={args.name}  dataset={args.dataset}  seed={seed}")
        print(f"  backbone : {args.init_from}")
        print(f"  out      : {out_dir}")
        print("=" * 70)
        cfg = FinetuneConfig(
            name=f"{args.name}_s{seed}",
            out_dir=str(out_dir),
            init_from=args.init_from,
            dataset=args.dataset,
            num_labels=args.num_labels,
            text_key=args.text_key,
            label_key=args.label_key,
            train_split=args.train_split,
            eval_split=args.eval_split,
            train_file=args.train_file,
            eval_file=args.eval_file,
            max_train=args.max_train,
            max_eval=args.max_eval,
            max_length=args.max_length,
            cache_dir=args.cache_dir,
            tokenizer=args.tokenizer,
            tokenizer_backend=args.tokenizer_backend,
            seed=seed,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            head_lr=args.head_lr,
            min_lr_frac=args.min_lr_frac,
            warmup_frac=args.warmup_frac,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            dropout=args.dropout,
            freeze_backbone=args.freeze_backbone,
            max_epochs=args.max_epochs,
            max_hours=args.max_hours,
            max_steps=args.max_steps,
            checkpoint_every_min=args.checkpoint_every_min,
            fresh=args.fresh,
            save_best=not args.no_save_best,
            log_interval=args.log_interval,
            eval_per_epoch=args.eval_per_epoch,
            eval_batch_size=args.eval_batch_size,
            device=args.device,
            dtype=args.dtype,
        )
        summaries.append(run_finetune(cfg))

    # -- aggregate across seeds ----------------------------------------------
    accs = [s["best"]["accuracy"] for s in summaries]
    f1s = [s["best"]["macro_f1"] for s in summaries]
    agg = {
        "name": args.name,
        "backbone_source": args.init_from,
        "dataset": args.dataset,
        "mode": "linear_probe" if args.freeze_backbone else "full_finetune",
        "seeds": args.seeds,
        "accuracy_mean": statistics.mean(accs),
        "accuracy_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
        "macro_f1_mean": statistics.mean(f1s),
        "macro_f1_std": statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
        "per_seed": [{"seed": s["seed"], "accuracy": s["best"]["accuracy"],
                      "macro_f1": s["best"]["macro_f1"]} for s in summaries],
    }
    agg_path = Path(args.out_root) / f"{args.name}_{ds_tag}_aggregate.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    print("\n" + "=" * 70)
    print(f"AGGREGATE  {args.name}  on  {args.dataset}  ({agg['mode']})")
    print("=" * 70)
    for r in agg["per_seed"]:
        print(f"  seed {r['seed']:<3} acc {r['accuracy']:.4f}  "
              f"macro-F1 {r['macro_f1']:.4f}")
    print(f"  {'mean':<8} acc {agg['accuracy_mean']:.4f} "
          f"+- {agg['accuracy_std']:.4f}   "
          f"macro-F1 {agg['macro_f1_mean']:.4f} +- {agg['macro_f1_std']:.4f}")
    if len(args.seeds) < 3:
        print("\n  NOTE: fewer than 3 seeds. Treat this as a smoke test, not a result.")
    print(f"\n  -> {agg_path}")
    print("  Compare models with: python scripts/benchmark.py "
          f"--finetune-root {args.out_root}")


if __name__ == "__main__":
    main()
