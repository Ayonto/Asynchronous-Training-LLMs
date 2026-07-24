#!/usr/bin/env python
"""Concatenate the training partitions into ONE .bin for the baseline model.

The merged model's shards each saw one partition (part_01.bin ... part_NN.bin).
The conventionally-trained control must see the SAME tokens, so this script
streams every partition into a single `part_all.bin`.

Why plain concatenation is enough
---------------------------------
`sap.data.ChunkSampler` draws a fresh random permutation of ALL chunks at every
epoch, so the byte order inside the file has no effect on the batch sequence.
There is no need to interleave or shuffle at the file level; doing so would only
cost a full random-access rewrite for zero benefit.

What this deliberately does NOT include
---------------------------------------
  * `val.bin`            — global held-out; no model may ever train on it
  * `part_XX.val.bin`    — per-partition held-out; same reason
  * `seed.bin`           — a *sample of* the partitions, so its tokens are
                           already present in part_*.bin. Including it would
                           silently upweight those documents in the baseline
                           and break the token match. (If your shards branched
                           from a seed, see --note-seed below.)

Seam effect: chunking is applied to the concatenated stream, so at each of the
N-1 joins one chunk of `seq_len` tokens straddles two partitions. That is N-1
mixed chunks out of millions — noted here for completeness, not a concern.

Examples
--------
# combine every part_NN.bin found in the dataset directory
python scripts/combine_partitions.py --data-dir data/fineweb_val

# explicit list + custom output name
python scripts/combine_partitions.py \
    --inputs data/fineweb_val/part_01.bin data/fineweb_val/part_02.bin \
    --out data/fineweb_val/part_12.bin
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sap.data import find_meta, load_tokens  # noqa: E402

CHUNK_TOKENS = 8_000_000  # streaming granularity (~16 MB per copy at uint16)


def discover_partitions(data_dir: Path) -> list[Path]:
    """All part_NN.bin in numeric order, excluding the *.val.bin held-out files."""
    pat = re.compile(r"^part_(\d+)\.bin$")
    found = []
    for p in data_dir.iterdir():
        m = pat.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    return [p for _, p in sorted(found)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir", help="directory holding part_01.bin ... part_NN.bin")
    src.add_argument("--inputs", nargs="+", help="explicit list of .bin files, in order")
    ap.add_argument("--out", default=None,
                    help="output .bin (default: <data-dir>/part_all.bin)")
    ap.add_argument("--dtype", default=None, choices=["uint16", "uint32"],
                    help="override the dtype from meta.json (rarely needed)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing output file")
    args = ap.parse_args()

    # -- resolve inputs -------------------------------------------------------
    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.is_dir():
            ap.error(f"not a directory: {data_dir}")
        parts = discover_partitions(data_dir)
        if not parts:
            ap.error(f"no part_NN.bin files found in {data_dir}")
    else:
        parts = [Path(p) for p in args.inputs]
        for p in parts:
            if not p.exists():
                ap.error(f"missing input: {p}")
        data_dir = parts[0].parent

    out_path = Path(args.out) if args.out else data_dir / "part_all.bin"
    if out_path.exists() and not args.force:
        ap.error(f"{out_path} already exists (pass --force to overwrite)")
    if out_path in parts:
        ap.error("output path collides with an input path")

    # -- dtype must be consistent; meta.json is the source of truth -----------
    meta = find_meta(parts[0])
    dtype = args.dtype or (meta["dtype"] if meta else None)
    if dtype is None:
        ap.error("no meta.json next to the inputs; pass --dtype explicitly")
    np_dtype = np.dtype(dtype)

    print(f"combining {len(parts)} partitions -> {out_path}  (dtype={dtype})")

    # -- stream, never load a whole partition into RAM ------------------------
    manifest_parts = []
    total = 0
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        with open(tmp_path, "wb") as fh:
            for p in parts:
                toks = load_tokens(p, dtype=dtype)
                n = len(toks)
                for lo in range(0, n, CHUNK_TOKENS):
                    np.asarray(toks[lo: lo + CHUNK_TOKENS], dtype=np_dtype).tofile(fh)
                manifest_parts.append({"path": str(p), "tokens": int(n),
                                       "offset": int(total)})
                total += n
                print(f"  {p.name:<24} {n:>15,} tokens  (running total {total:,})")
                del toks
        tmp_path.replace(out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    # -- provenance: never mutate the dataset's own meta.json -----------------
    manifest = {
        "output": str(out_path),
        "dtype": dtype,
        "total_tokens": int(total),
        "num_partitions": len(parts),
        "sources": manifest_parts,
        "note": "Union of the shard training partitions; excludes val.bin, "
                "part_XX.val.bin and seed.bin by design.",
    }
    man_path = out_path.with_suffix(".manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    size_gb = out_path.stat().st_size / 1e9
    print(f"\ndone: {total:,} tokens ({size_gb:.2f} GB) -> {out_path}")
    print(f"manifest: {man_path}")
    if meta is None:
        print("\nWARNING: no meta.json in this directory. The trainer needs one "
              "(or pass --data-dtype) to know the token dtype.")
    else:
        print(f"meta.json present ({data_dir / 'meta.json'}) — the trainer will "
              "pick up dtype/vocab automatically.")
    print("\nUse this token count to match the baseline budget to the shards:")
    print(f"  --match-tokens {total}    (one full epoch over the union)")


if __name__ == "__main__":
    main()
