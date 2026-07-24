#!/usr/bin/env python
"""Lightweight progress viewer for all shards — reads only the tiny
status.json files the trainer refreshes (no torch import, instant, zero load
on training).

    python scripts/progress.py --runs-dir runs            # print once, exit
    python scripts/progress.py --runs-dir runs --watch 30 # refresh every 30s

States: running / completed / stopped (graceful pause) / failed (reason shown)
/ stale? (marked running but not updated recently — the process likely died
without cleanup; rerun its training command to resume).
"""

import argparse
import json
import sys
import time
from pathlib import Path


def fmt_tokens(n):
    if n is None:
        return "-"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return f"{n:,}"


def collect(runs_dir: Path):
    rows = []
    for status in sorted(runs_dir.glob("*/status.json")):
        try:
            s = json.loads(status.read_text())
        except Exception:  # noqa: BLE001 — mid-write or corrupt; skip this cycle
            continue
        age = time.time() - s.get("updated", 0)
        state = s.get("state", "?")
        if state == "running" and age > 300:
            state = "stale?"   # says running but silent >5 min: process likely dead
        rows.append({
            "name": s.get("name", status.parent.name),
            "state": state,
            "step": s.get("step"),
            "epoch": s.get("epoch"),
            "tokens": s.get("tokens_seen"),
            "loss": s.get("loss"),
            "val_ppl": s.get("val_ppl"),
            "elapsed_h": s.get("elapsed_hours"),
            "progress": s.get("progress"),
            "eta_h": s.get("eta_hours"),
            "age_s": age,
            "error": s.get("error"),
        })
    return rows


def render(rows) -> str:
    if not rows:
        return "no status.json files found — has any shard started training?"
    hdr = (f"{'shard':<12}{'state':<11}{'step':>9}{'ep':>4}{'tokens':>9}"
           f"{'loss':>8}{'val_ppl':>9}{'hours':>7}{'prog':>7}{'ETA(h)':>8}  updated")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        prog = f"{100 * r['progress']:.1f}%" if r["progress"] is not None else "-"
        eta = f"{r['eta_h']:.1f}" if r["eta_h"] is not None else "-"
        loss = f"{r['loss']:.3f}" if r["loss"] is not None else "-"
        ppl = f"{r['val_ppl']:.1f}" if r["val_ppl"] is not None else "-"
        upd = f"{r['age_s']:.0f}s ago" if r["age_s"] < 120 else f"{r['age_s'] / 60:.0f}m ago"
        lines.append(
            f"{r['name']:<12}{r['state']:<11}{(r['step'] or 0):>9,}{(r['epoch'] or 0):>4}"
            f"{fmt_tokens(r['tokens']):>9}{loss:>8}{ppl:>9}"
            f"{(r['elapsed_h'] or 0):>7.2f}{prog:>7}{eta:>8}  {upd}"
        )
    for r in rows:
        if r["error"]:
            lines.append(f"\n{r['name']} FAILED: {r['error']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="runs",
                    help="directory containing one subfolder per shard")
    ap.add_argument("--watch", type=float, default=None,
                    help="refresh every N seconds (default: print once and exit)")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if args.watch is None:
        print(render(collect(runs_dir)))
        return
    try:
        while True:
            sys.stdout.write("\x1b[2J\x1b[H")   # clear screen, cursor home
            print(f"SAP training progress — {time.strftime('%H:%M:%S')} "
                  f"(refresh {args.watch:.0f}s, Ctrl+C to exit; training is unaffected)\n")
            print(render(collect(runs_dir)), flush=True)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
