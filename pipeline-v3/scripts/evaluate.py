#!/usr/bin/env python
"""Dynamic evaluation harness: any set of models x any set of token files.

Hand it shards, the seed, merged models, and baselines together with val.bin
(and per-partition vals if you made them) and it prints the full results
grid — loss, perplexity, next-token accuracy — plus parameter counts and
token provenance for every model. Results are also written as JSON.

Examples
--------
# the standard thesis table: all parents + merged + baseline on global val:
python scripts/evaluate.py \
    --models runs/shard_01/final.pt runs/shard_02/final.pt runs/shard_03/final.pt \
             runs/merged_123.pt runs/avg_123.pt \
    --data data/mycorpus/val.bin --out results/main.json

# specialty retention: merged model on each partition's held-out set:
python scripts/evaluate.py --models runs/merged_123.pt \
    --data data/mycorpus/part_01.val.bin data/mycorpus/part_02.val.bin \
           data/mycorpus/part_03.val.bin

# qualitative sample:
python scripts/evaluate.py --models runs/merged_123.pt --generate \
    --tokenizer gpt2 --prompt "The history of mathematics" --max-new-tokens 150
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sap.evaluate import (  # noqa: E402
    evaluate_models,
    format_results_table,
    generate_text,
    save_results,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="checkpoint paths")
    p.add_argument("--data", nargs="+", default=[], help="token .bin files to evaluate on")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-batches", type=int, default=None,
                   help="cap batches per (model, file) pair; default = whole file")
    p.add_argument("--device", default="auto")
    p.add_argument("--fp32", action="store_true", help="disable bf16 autocast")
    p.add_argument("--data-dtype", default=None, choices=[None, "uint16", "uint32"])
    p.add_argument("--out", default=None, help="write results JSON here")
    # generation
    p.add_argument("--generate", action="store_true")
    p.add_argument("--tokenizer", default=None, help="required with --generate")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()

    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device

    if args.data:
        results = evaluate_models(
            model_paths=args.models,
            bin_paths=args.data,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            device=device,
            fp32=args.fp32,
            dtype=args.data_dtype,
        )
        print()
        print(format_results_table(results))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            save_results(results, args.out)
            print(f"\nresults JSON -> {args.out}")

    if args.generate:
        if not args.tokenizer:
            p.error("--generate requires --tokenizer (name or path)")
        for mp in args.models:
            print(f"\n=== sample from {mp} ===")
            text = generate_text(
                mp, args.tokenizer, args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k, device=device,
            )
            print(text)

    if not args.data and not args.generate:
        p.error("nothing to do: pass --data files and/or --generate")


if __name__ == "__main__":
    main()
