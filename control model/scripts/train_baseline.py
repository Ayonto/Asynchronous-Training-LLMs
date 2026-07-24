#!/usr/bin/env python
"""Train the conventional control model: ONE model, ONE run, ALL the data.

This is baseline #4 of the framework report (section 13.3) — "the full-data model at
equal total compute", the number the whole thesis is measured against. It is
deliberately independent of the merge pipeline: it never imports sap.merge and
never reads a shard. It shares only the model definition and the trainer, so
that any difference in the results comes from the training paradigm and not
from an incidental difference in architecture or optimizer.

Crash safety, budgets and resume are inherited from sap.train unchanged:
rerunning the SAME command resumes from out_dir/latest.pt and reproduces the
exact batch sequence the interrupted run would have seen.

Making the comparison fair
--------------------------
Two knobs must be matched deliberately, and they are matched by DIFFERENT flags:

  architecture  --match-merged runs/merged/merged.pt
                Reads the merged model's per-layer (n_heads, d_ff) out of the
                checkpoint and builds a from-scratch model with exactly those
                widths. The control becomes an architectural twin of the merged
                model: same layers, same width, same parameter count — the only
                difference is that it was trained conventionally.

  token budget  --match-tokens-from runs/shard_*/final.pt
                Sums `tokens_seen` across the shards and converts it into a step
                budget, so the control consumes the same number of training
                tokens the shards collectively consumed.

Report BOTH of the following, because they answer different questions:
  * equal tokens   — "is merging as good as one run over the same data?"
  * equal params   — "is a 546M merged model as good as a real 546M model?"
The flags above give you both at once; see BASELINE_GUIDE.md section 4.

Examples
--------
# architectural twin of the merged model, same token budget as the 5 shards
python scripts/train_baseline.py \
    --name baseline_546m --family configs/family_reference.json \
    --data data/fineweb_val/part_all.bin --val data/fineweb_val/val.bin \
    --out-dir runs/baseline_546m \
    --match-merged runs/merged/merged.pt \
    --match-tokens-from runs/shard_01/final.pt runs/shard_02/final.pt \
                        runs/shard_03/final.pt runs/shard_04/final.pt \
                        runs/shard_05/final.pt \
    --batch-size 4 --grad-accum 16 --checkpoint-every-min 30

# just show what would be trained, then exit
python scripts/train_baseline.py ... --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Reduce CUDA allocator fragmentation; must be set before torch initializes.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from sap.config import FamilyConfig, ModelWidths, merged_widths  # noqa: E402
from sap.model import load_checkpoint  # noqa: E402
from sap.train import TrainConfig, run_training  # noqa: E402


# ---------------------------------------------------------------------------
# Width resolution — never hardcode the target size, always derive it
# ---------------------------------------------------------------------------

def widths_from_checkpoints(paths: list[str], family: FamilyConfig) -> ModelWidths:
    """Per-layer width sum across N shard checkpoints (== the merged widths)."""
    ws = []
    for p in paths:
        ck = load_checkpoint(p)
        fam = FamilyConfig.from_dict(ck["family"])
        if fam.to_dict() != family.to_dict():
            raise ValueError(
                f"{p} belongs to a different family than {family.to_dict()}"
            )
        ws.append(ModelWidths.from_dict(ck["widths"]))
    return merged_widths(ws)


def widths_of_checkpoint(path: str, family: FamilyConfig) -> ModelWidths:
    ck = load_checkpoint(path)
    fam = FamilyConfig.from_dict(ck["family"])
    if fam.to_dict() != family.to_dict():
        raise ValueError(f"{path} belongs to a different family than {family.to_dict()}")
    return ModelWidths.from_dict(ck["widths"])


def spec(values: list[int]) -> str:
    """Render a per-layer list for TrainConfig's comma-list spec format,
    collapsing to a single int when every layer agrees."""
    return str(values[0]) if len(set(values)) == 1 else ",".join(str(v) for v in values)


def count_params(family: FamilyConfig, widths: ModelWidths) -> dict:
    """Closed-form parameter count — avoids materializing a multi-hundred-MB
    model just to print a number."""
    E = family.vocab_size * family.d_model          # tied embedding == LM head
    stackable = 0
    for h, f in zip(widths.n_heads, widths.d_ff):
        stackable += 4 * family.d_model * family.d_head * h   # wq,wk,wv,wo
        stackable += 3 * family.d_model * f                   # w_gate,w_up,w_down
        stackable += 2 * family.d_model                       # two RMSNorm gains
    stackable += family.d_model                               # final norm
    return {"embedding": E, "stackable": stackable, "total": E + stackable}


def sum_tokens(paths: list[str]) -> int:
    total = 0
    for p in paths:
        ck = load_checkpoint(p)
        t = ck.get("meta", {}).get("tokens_seen")
        if not t:
            raise ValueError(f"{p} has no meta.tokens_seen — cannot match the budget")
        total += int(t)
        print(f"    {Path(p).parent.name:<20} {t:>18,} tokens")
    return total


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # identity / io
    p.add_argument("--name", default="baseline", help="model name (goes into meta)")
    p.add_argument("--family", required=True, help="path to the family JSON")
    p.add_argument("--data", required=True,
                   help="combined .bin from scripts/combine_partitions.py")
    p.add_argument("--val", default=None, help="held-out .bin for periodic evals")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--data-dtype", default=None, choices=["uint16", "uint32"])

    # architecture — pick exactly one
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--match-merged", metavar="CKPT",
                   help="build the control with the merged model's exact widths")
    g.add_argument("--match-shards", nargs="+", metavar="CKPT",
                   help="build the control with the SUM of these shards' widths")
    g.add_argument("--n-heads", help="explicit: int or per-layer comma list")
    p.add_argument("--d-ff", help="explicit: int or per-layer comma list "
                                  "(required with --n-heads)")

    # token budget matching
    p.add_argument("--match-tokens", type=int, default=None,
                   help="train for this many tokens (converted to --max-steps)")
    p.add_argument("--match-tokens-from", nargs="+", metavar="CKPT", default=None,
                   help="sum meta.tokens_seen over these shard checkpoints and "
                        "train for that many tokens")

    # optimization
    p.add_argument("--init-seed", type=int, default=7777)
    p.add_argument("--fresh", action="store_true",
                   help="ignore an existing latest.pt and restart from step 0")
    p.add_argument("--batch-size", type=int, default=8)
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

    # checkpointing
    p.add_argument("--checkpoint-every-min", type=float, default=30.0)
    p.add_argument("--checkpoint-every-steps", type=int, default=None)
    p.add_argument("--keep-history", type=int, default=2,
                   help="step-stamped snapshots to retain besides latest.pt")

    # logging / eval
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=20)

    # hardware
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp32"], default="auto")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--sdpa", choices=["auto", "math"], default="auto")
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="print the resolved plan and exit without training")
    args = p.parse_args()

    family = FamilyConfig.from_json(args.family)

    # -- resolve widths -------------------------------------------------------
    reference = None
    if args.match_merged:
        widths = widths_of_checkpoint(args.match_merged, family)
        reference = ("merged checkpoint", args.match_merged)
    elif args.match_shards:
        widths = widths_from_checkpoints(args.match_shards, family)
        reference = (f"sum of {len(args.match_shards)} shards", ", ".join(args.match_shards))
    else:
        if not args.d_ff:
            p.error("--n-heads requires --d-ff")
        widths = ModelWidths.from_spec(family.n_layers, args.n_heads, args.d_ff)

    if widths.n_layers != family.n_layers:
        p.error(f"widths describe {widths.n_layers} layers, family has {family.n_layers}")

    counts = count_params(family, widths)

    # -- resolve the token budget --------------------------------------------
    seq_len = args.seq_len or family.max_seq_len
    tokens_per_step = args.batch_size * args.grad_accum * seq_len
    max_steps = args.max_steps
    target_tokens = args.match_tokens

    if args.match_tokens_from:
        print("matching the shards' total token budget:")
        summed = sum_tokens(args.match_tokens_from)
        target_tokens = (target_tokens or 0) + summed if args.match_tokens else summed

    if target_tokens is not None:
        derived = -(-target_tokens // tokens_per_step)   # ceil
        if max_steps is None:
            max_steps = derived
        else:
            print(f"NOTE: --max-steps {max_steps} given explicitly; the token match "
                  f"would have used {derived}")

    if args.max_epochs is None and args.max_hours is None and max_steps is None:
        p.error("set a budget: --match-tokens / --match-tokens-from / "
                "--max-epochs / --max-hours / --max-steps")

    # -- report the plan ------------------------------------------------------
    print("\n" + "=" * 70)
    print("BASELINE PLAN — conventionally trained control model")
    print("=" * 70)
    print(f"  family        : {Path(args.family).name}  "
          f"(L={family.n_layers}, d_model={family.d_model}, d_head={family.d_head}, "
          f"vocab={family.vocab_size})")
    if reference:
        print(f"  widths from   : {reference[0]}")
        print(f"                  {reference[1]}")
    print(f"  n_heads/layer : {spec(widths.n_heads)}")
    print(f"  d_ff/layer    : {spec(widths.d_ff)}")
    print(f"  parameters    : {counts['total'] / 1e6:.1f}M "
          f"({counts['stackable'] / 1e6:.1f}M stackable + "
          f"{counts['embedding'] / 1e6:.1f}M embedding)")
    print(f"  data          : {args.data}")
    print(f"  seq_len       : {seq_len}")
    print(f"  tokens/step   : {tokens_per_step:,} "
          f"(batch {args.batch_size} x accum {args.grad_accum} x seq {seq_len})")
    if target_tokens:
        print(f"  token target  : {target_tokens:,}")
    print(f"  max_steps     : {max_steps}")
    print(f"  max_epochs    : {args.max_epochs}")
    print(f"  max_hours     : {args.max_hours}")
    print(f"  checkpoints   : every {args.checkpoint_every_min} min -> "
          f"{args.out_dir}/latest.pt  (resume by rerunning this command)")

    if args.match_merged:
        # Independent cross-check: count the merged checkpoint's ACTUAL tensors
        # rather than re-printing the formula's own output. This catches both a
        # bug in count_params and a checkpoint whose widths metadata disagrees
        # with its weights.
        ck = load_checkpoint(args.match_merged)
        real = sum(t.numel() for t in ck["model_state"].values())
        print(f"\n  MATCH CHECK — control vs merged model")
        print(f"    control params (from widths) : {counts['total'] / 1e6:>9.2f}M")
        print(f"    merged  params (from tensors): {real / 1e6:>9.2f}M")
        delta = counts["total"] - real
        if delta == 0:
            print("    -> exact match: the control is an architectural twin.")
        else:
            print(f"    -> DIFFER by {delta:+,} params ({delta / max(1, real):+.3%}).")
            print("       Expected only if the merged checkpoint ties/unties weights")
            print("       differently. Investigate before trusting the comparison.")
        del ck

    # memory warning: the merged model is normally never trained, so this size
    # is a new regime for the training machine
    approx_optim_gb = counts["total"] * 16 / 1e9   # fp32 param+grad+2 Adam moments
    print(f"\n  ESTIMATED optimizer-state memory: ~{approx_optim_gb:.1f} GB "
          f"(params+grads+Adam, fp32 master)")
    print("  Activations are extra and scale with batch_size x seq_len. If you OOM,")
    print("  lower --batch-size and raise --grad-accum by the same factor: the")
    print("  optimizer batch, and therefore the training math, is unchanged.")
    print("=" * 70 + "\n")

    if args.dry_run:
        print("--dry-run: exiting without training.")
        return

    if not torch.cuda.is_available() and args.device in ("auto", "cuda"):
        print("WARNING: no CUDA device visible — this will train on CPU and will "
              "be impractically slow at this model size.\n")

    cfg = TrainConfig(
        name=args.name,
        out_dir=args.out_dir,
        data_path=args.data,
        val_path=args.val,
        data_dtype=args.data_dtype,
        family=family,
        n_heads_spec=spec(widths.n_heads),
        d_ff_spec=spec(widths.d_ff),
        init_from="scratch",          # the control is trained conventionally
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
        max_steps=max_steps,
        checkpoint_every_min=args.checkpoint_every_min,
        checkpoint_every_steps=args.checkpoint_every_steps,
        keep_history=args.keep_history,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        device=args.device,
        dtype=args.dtype,
        compile=args.compile,
        sdpa_backend=args.sdpa,
        pin_memory=not args.no_pin_memory,
    )
    summary = run_training(cfg)
    print(f"\nbaseline complete: {summary['tokens_seen']:,} tokens in "
          f"{summary['elapsed_hours']:.2f}h -> {summary['final_checkpoint']}")
    print("Next: evaluate it head-to-head with the merged model,")
    print(f"  python scripts/benchmark.py --models {summary['final_checkpoint']} "
          f"runs/merged/merged.pt --bins {args.val or 'data/<set>/val.bin'}")


if __name__ == "__main__":
    main()
