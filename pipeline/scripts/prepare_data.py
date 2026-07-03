#!/usr/bin/env python
"""Tokenize any text corpus and partition it into D_1..D_N + validation splits.

Examples
--------
# 5 random partitions of a local jsonl corpus, 0.5% global val, 5% seed sample:
python scripts/prepare_data.py \
    --inputs corpus.jsonl --tokenizer gpt2 --out-dir data/mycorpus \
    --num-partitions 5 --val-fraction 0.005 --seed-fraction 0.05

# a streamed HuggingFace dataset, capped at 2M documents:
python scripts/prepare_data.py \
    --inputs hf:HuggingFaceFW/fineweb-edu:sample-10BT:train \
    --tokenizer gpt2 --out-dir data/fineweb --num-partitions 5 --max-docs 2000000

# domain partitioning: each input file becomes its own partition:
python scripts/prepare_data.py \
    --inputs code.jsonl prose.jsonl math.jsonl --partition-mode by-input \
    --num-partitions 3 --tokenizer gpt2 --out-dir data/domains
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sap.data import prepare_dataset  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True,
                   help=".txt / .jsonl files or hf:name[:config][:split] specs")
    p.add_argument("--tokenizer", required=True,
                   help="HuggingFace tokenizer name or local path (family skeleton!)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-partitions", type=int, required=True)
    p.add_argument("--val-fraction", type=float, default=0.005,
                   help="global held-out validation share (never trained on)")
    p.add_argument("--seed-fraction", type=float, default=0.0,
                   help="share of training docs ALSO copied into seed.bin "
                        "(mixed sample for the optional seed phase); 0 = no seed.bin")
    p.add_argument("--partition-val-fraction", type=float, default=0.0,
                   help="per-partition held-out share (part_XX.val.bin) for "
                        "'does the merged model keep each shard's specialty' evals")
    p.add_argument("--partition-mode", choices=["random", "blocks", "by-input"],
                   default="random",
                   help="random: i.i.d. doc-level split; blocks: contiguous blocks of "
                        "docs round-robin (domain-ish); by-input: input file k -> "
                        "partition k (true domain split)")
    p.add_argument("--block-docs", type=int, default=10_000)
    p.add_argument("--routing-seed", type=int, default=1234)
    p.add_argument("--text-key", default="text")
    p.add_argument("--txt-mode", choices=["line", "file"], default="line")
    p.add_argument("--eos-id", type=int, default=None,
                   help="override document-separator token id if the tokenizer has no EOS")
    p.add_argument("--max-docs", type=int, default=None)
    args = p.parse_args()

    meta = prepare_dataset(
        inputs=args.inputs,
        tokenizer_name=args.tokenizer,
        out_dir=args.out_dir,
        num_partitions=args.num_partitions,
        val_fraction=args.val_fraction,
        seed_fraction=args.seed_fraction,
        partition_val_fraction=args.partition_val_fraction,
        mode=args.partition_mode,
        block_docs=args.block_docs,
        seed=args.routing_seed,
        text_key=args.text_key,
        txt_mode=args.txt_mode,
        eos_id=args.eos_id,
        max_docs=args.max_docs,
    )

    print("\n=== dataset prepared ===")
    for name, info in meta["files"].items():
        print(f"  {name:<22} {info['tokens']:>14,} tokens  {info['documents']:>10,} docs")
    print(f"\nvocab_size = {meta['vocab_size']}  (dtype {meta['dtype']})")
    print("\nFamily config must use this vocab size. Suggested snippet:")
    print(json.dumps({"vocab_size": meta["vocab_size"]}, indent=2))


if __name__ == "__main__":
    main()
