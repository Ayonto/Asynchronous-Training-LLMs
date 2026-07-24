#!/usr/bin/env python
"""Data / tokenizer integrity check for a prepared SAP data directory.

Answers, with measurements instead of guesses:
  * are all token ids in every .bin inside the vocabulary? (an out-of-range id
    is the ONE way bad data can hard-crash training: it indexes the embedding
    table out of bounds inside native code)
  * do the files have sane sizes, EOS density, and id distributions?
  * does the data decode back to real text? (--decode, needs tiktoken)

Usage:
    python scripts/check_data.py --data-dir data/fineweb_val
    python scripts/check_data.py --data-dir data/fineweb_val --decode 2
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402


def scan_bin(path: Path, dtype: str, vocab_size: int, eos_id: int, chunk: int = 64_000_000):
    toks = np.memmap(str(path), dtype=np.dtype(dtype), mode="r")
    n = len(toks)
    mx, mn = -1, 1 << 60
    n_invalid = 0
    n_eos = 0
    for i in range(0, n, chunk):
        c = toks[i: i + chunk]
        mx = max(mx, int(c.max()))
        mn = min(mn, int(c.min()))
        n_invalid += int((c >= vocab_size).sum())
        n_eos += int((c == eos_id).sum())
    return {
        "tokens": n,
        "min_id": mn,
        "max_id": mx,
        "invalid_ids": n_invalid,           # MUST be 0
        "eos_count": n_eos,
        "avg_doc_len": (n / n_eos) if n_eos else float("inf"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--decode", type=int, default=0,
                    help="decode this many sample documents per training partition "
                         "to visually confirm the data is real text (needs tiktoken)")
    args = ap.parse_args()

    d = Path(args.data_dir)
    meta_path = d / "meta.json"
    if not meta_path.exists():
        sys.exit(f"ERROR: {meta_path} not found — is this a prepared data directory?")
    meta = json.loads(meta_path.read_text())
    vocab, dtype = meta["vocab_size"], meta["dtype"]
    eos = meta.get("eos_id", -1)   # -1 matches nothing if meta predates eos_id
    print(f"meta.json: tokenizer={meta.get('tokenizer', '?')}  vocab={vocab}  "
          f"dtype={dtype}  eos_id={eos}\n")

    bins = sorted(d.glob("*.bin"))
    if not bins:
        sys.exit("ERROR: no .bin files found")

    all_ok = True
    print(f"{'file':<22}{'tokens':>15}{'min':>7}{'max':>8}{'invalid':>9}"
          f"{'eos%':>7}{'avg doc':>9}  verdict")
    print("-" * 86)
    for b in bins:
        r = scan_bin(b, dtype, vocab, eos)
        ok = r["invalid_ids"] == 0 and r["max_id"] < vocab and r["min_id"] >= 0
        all_ok &= ok
        print(f"{b.name:<22}{r['tokens']:>15,}{r['min_id']:>7}{r['max_id']:>8}"
              f"{r['invalid_ids']:>9}{100 * r['eos_count'] / max(r['tokens'], 1):>6.2f}%"
              f"{r['avg_doc_len']:>9.0f}  {'OK' if ok else '*** CORRUPT ***'}")

    if args.decode > 0:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        for b in bins:
            if not b.name.startswith("part_") or b.name.endswith(".val.bin"):
                continue
            toks = np.memmap(str(b), dtype=np.dtype(dtype), mode="r")
            print(f"\n--- sample from {b.name} ---")
            pos = 0
            for _ in range(args.decode):
                end = pos
                while end < min(pos + 2000, len(toks)) and toks[end] != eos:
                    end += 1
                ids = [int(t) for t in toks[pos:end]]
                text = enc.decode(ids)
                print(f"  {text[:300]!r}")
                pos = end + 1

    print("\n" + ("ALL FILES CLEAN: every token id is inside the vocabulary. "
                  "The data cannot crash training via out-of-range indexing."
                  if all_ok else
                  "*** CORRUPTION DETECTED: some ids are outside the vocabulary. "
                  "DO NOT train on this data — re-run preparation. ***"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
