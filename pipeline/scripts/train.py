#!/usr/bin/env python
"""Train one model (seed or shard) on one .bin file.

Crash-safe by default: rerunning the SAME command auto-resumes from
out_dir/latest.pt. Use --fresh to deliberately restart.

Examples
--------
# optional seed phase (small budget on the mixed sample):
python scripts/train.py --name seed --family configs/family.json \
    --data data/mycorpus/seed.bin --val data/mycorpus/val.bin \
    --out-dir runs/seed --n-heads 4 --d-ff 1024 --max-epochs 1

# shard 1, branched from the seed, checkpoint every hour, 2 epochs:
python scripts/train.py --name shard_01 --family configs/family.json \
    --data data/mycorpus/part_01.bin --val data/mycorpus/val.bin \
    --out-dir runs/shard_01 --init-from runs/seed/final.pt \
    --checkpoint-every-min 60 --max-epochs 2

# shard 2, no seed (independent from scratch), time-limited, ckpt every 3h:
python scripts/train.py --name shard_02 --family configs/family.json \
    --data data/mycorpus/part_02.bin --out-dir runs/shard_02 \
    --n-heads 4 --d-ff 1024 --init-seed 2002 \
    --checkpoint-every-min 180 --max-hours 12
"""

import argparse
import os
import sys
from pathlib import Path

# Reduce CUDA allocator fragmentation; must be set before torch initializes.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sap.config import FamilyConfig  # noqa: E402
from sap.train import TrainConfig, run_training  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # identity / io
    p.add_argument("--name", required=True, help="model name (seed, shard_01, ...)")
    p.add_argument("--family", required=True, help="path to the family JSON")
    p.add_argument("--data", required=True, help="training partition .bin")
    p.add_argument("--val", default=None, help="optional validation .bin for quick evals")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--data-dtype", default=None, choices=[None, "uint16", "uint32"])
    # widths / init
    p.add_argument("--n-heads", default=None,
                   help="int or per-layer comma list; required from scratch, "
                        "omit with --init-from to inherit the checkpoint's widths")
    p.add_argument("--d-ff", default=None,
                   help="int or per-layer comma list; same rules as --n-heads")
    p.add_argument("--init-from", default="scratch",
                   help="'scratch' or a checkpoint path (seed branch / grow)")
    p.add_argument("--init-seed", type=int, default=1337,
                   help="weight-init RNG seed; give each from-scratch shard its own")
    p.add_argument("--fresh", action="store_true",
                   help="ignore an existing latest.pt and restart from step 0")
    # optimization
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--schedule", choices=["cosine", "constant"], default="cosine")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--data-seed", type=int, default=42)
    # budget (whichever hits first stops the run)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    # checkpoint cadence (per model)
    p.add_argument("--checkpoint-every-min", type=float, default=60.0)
    p.add_argument("--checkpoint-every-steps", type=int, default=None)
    p.add_argument("--keep-history", type=int, default=0)
    # logging / eval
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=20)
    # hardware
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp32"], default="auto")
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    if args.max_epochs is None and args.max_hours is None and args.max_steps is None:
        p.error("set at least one budget: --max-epochs, --max-hours, or --max-steps")

    cfg = TrainConfig(
        name=args.name,
        out_dir=args.out_dir,
        data_path=args.data,
        val_path=args.val,
        data_dtype=args.data_dtype,
        family=FamilyConfig.from_json(args.family),
        n_heads_spec=args.n_heads,
        d_ff_spec=args.d_ff,
        init_from=args.init_from,
        init_seed=args.init_seed,
        fresh=args.fresh,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        seq_len=args.seq_len,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        schedule=args.schedule,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        data_seed=args.data_seed,
        max_epochs=args.max_epochs,
        max_hours=args.max_hours,
        max_steps=args.max_steps,
        checkpoint_every_min=args.checkpoint_every_min,
        checkpoint_every_steps=args.checkpoint_every_steps,
        keep_history=args.keep_history,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        device=args.device,
        dtype=args.dtype,
        compile=args.compile,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
