#!/usr/bin/env python
"""Single-file SAP validation runner — NO seed. Run once and walk away.

    python validation/validate.py

On one GPU, within a 12-hour budget, this does everything hands-off:

  1. PREPARE  ~4B tokens of FineWeb-Edu into `num_shards` random partitions
              (+ held-out val). Skipped automatically if already prepared.
  2. TRAIN    each shard from scratch, fully independent (no seed, its own
              init), back-to-back — the next starts the instant the previous
              finishes. Remaining time is divided evenly across the shards so
              the whole run fits the budget.
  3. MERGE    the shards three ways: scaled (exact average — the method),
              unscaled (exact sum), and a naive weight-average baseline.
              Each stack merge is gated: it must be provably exact or it is
              not saved.
  4. EVALUATE every model on held-out val -> loss / perplexity / accuracy grid.

Reboot-safe: if the machine restarts, run the same command again. Finished
shards are skipped, an interrupted shard resumes from its last checkpoint, and
the run continues. Nothing is retrained, nothing is lost.

Everything is configured in CONFIG below (edit in place) or via an optional
JSON override: `python validation/validate.py --config my.json`.

This file changes NO merge math — it calls the same gated `merge_checkpoints`
the CLI uses.
"""

import argparse
import datetime
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from sap.config import FamilyConfig  # noqa: E402
from sap.data import find_meta  # noqa: E402
from sap.evaluate import evaluate_models, format_results_table, save_results  # noqa: E402
from sap.merge import merge_checkpoints  # noqa: E402
from sap.model import load_checkpoint  # noqa: E402
from sap.train import TrainConfig, run_training  # noqa: E402


# ===========================================================================
# CONFIG — edit these. Every path is relative to the repo root.
# ===========================================================================
CONFIG = {
    # ---- data preparation -------------------------------------------------
    "hf_dataset": "hf:HuggingFaceFW/fineweb-edu:sample-10BT:train",
    "tokenizer": "gpt2",                     # family tokenizer (skeleton)
    "tokenizer_backend": "tiktoken",         # 'tiktoken' (stable) or 'hf'; identical gpt2 ids

    "data_dir": "data/fineweb_val",
    "num_shards": 5,                         # partitions == shards (no seed)
    "prepare_max_tokens": 4_000_000_000,     # ~4B: plenty for 5 shards x ~2h, quick to prep
    "val_fraction": 0.005,                   # global held-out (never trained on)
    "partition_val_fraction": 0.001,         # per-partition held-out (specialty evals)

    # ---- family + shard width (same as the real 1B run) -------------------
    "family": "configs/family_reference.json",
    "n_heads": "4",                          # per shard  -> ~171M each
    "d_ff": "512",

    # ---- time budget ------------------------------------------------------
    "total_hours": 12.0,                     # your access window (prep + train + merge + eval)
    "reserve_hours": 0.5,                    # kept aside for merge + eval
    "min_shard_seconds": 300,                # if less than this remains, stop training

    # ---- training ---------------------------------------------------------
    "seq_len": 1024,
    "batch_size": 16,                        # lower to 12/8 if you hit OOM
    "grad_accum": 8,
    "compile": False,                        # torch.compile/Triton is fragile on freshly
                                             # patched GPU drivers; turn on only once stable
    "checkpoint_every_min": 20,              # crash-safety cadence per shard
    "log_interval": 20,
    "eval_interval": 400,                    # quick val eval during training
    "eval_batches": 20,
    "shard_init_seed_base": 1000,            # shard k gets init_seed base + k (independent inits)

    # ---- final evaluation -------------------------------------------------
    "eval_max_batches": 200,                 # cap per (model, file); None = whole file

    # ---- outputs ----------------------------------------------------------
    "runs_dir": "validation/runs",
    "results_dir": "validation/results",

    # ---- merges to produce ------------------------------------------------
    "merges": [
        {"name": "merged_scaled",   "method": "stack", "scaled": True,  "alpha_mode": "tokens"},
        {"name": "merged_unscaled", "method": "stack", "scaled": False, "alpha_mode": "tokens"},
        {"name": "baseline_avg",    "method": "avg",   "alpha_mode": "uniform"},
    ],
}
# ===========================================================================


def rel(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO / p)


class Log:
    """Timestamped logger: prints AND appends to progress.log."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def prepare_if_needed(cfg: dict, family: FamilyConfig, data_dir: Path, log: Log) -> None:
    """Tokenize + partition the corpus, unless it is already done.

    meta.json is written only after a full successful prep, so its presence is
    the completion sentinel — a prep interrupted by a crash re-runs cleanly."""
    if find_meta(data_dir / "val.bin") is not None and (data_dir / "meta.json").exists():
        meta = json.load(open(data_dir / "meta.json", encoding="utf-8"))
        log(f"data already prepared in {data_dir} "
            f"({meta.get('total_tokens', '?'):,} tokens); skipping prep")
    else:
        try:
            import transformers  # noqa: F401
            import datasets       # noqa: F401
        except Exception as e:  # noqa: BLE001
            log(f"FATAL: data prep needs `transformers` and `datasets` installed ({e}). "
                "Install them, or pre-prepare the data with scripts/prepare_data.py.")
            sys.exit(1)
        from sap.data import prepare_dataset
        log(f"preparing ~{cfg['prepare_max_tokens'] / 1e9:.1f}B tokens from "
            f"{cfg['hf_dataset']} into {cfg['num_shards']} partitions (needs network)...")
        meta = prepare_dataset(
            inputs=[cfg["hf_dataset"]],
            tokenizer_name=cfg["tokenizer"],
            out_dir=str(data_dir),
            num_partitions=cfg["num_shards"],
            tokenizer_backend=cfg.get("tokenizer_backend", "hf"),
            val_fraction=cfg["val_fraction"],
            seed_fraction=0.0,                       # NO seed this run
            partition_val_fraction=cfg["partition_val_fraction"],
            mode="random",
            max_tokens=cfg["prepare_max_tokens"],
        )
        log(f"prep done: {meta['total_tokens']:,} tokens across "
            f"{len(meta['files'])} files")

    if meta["vocab_size"] != family.vocab_size:
        log(f"FATAL: tokenizer vocab {meta['vocab_size']} != family vocab "
            f"{family.vocab_size}. Fix vocab_size in {cfg['family']} (the tokenizer "
            "is part of the family skeleton).")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="optional JSON file whose keys override CONFIG")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg.update(json.load(f))

    results_dir = rel(cfg["results_dir"]); results_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = rel(cfg["runs_dir"]); runs_dir.mkdir(parents=True, exist_ok=True)
    data_dir = rel(cfg["data_dir"])
    log = Log(results_dir / "progress.log")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    family = FamilyConfig.from_json(rel(cfg["family"]))

    log(f"=== SAP validation (no seed) ===  device={device}  "
        f"budget={cfg['total_hours']}h  shards={cfg['num_shards']}")
    if device == "cpu":
        log("WARNING: no CUDA detected — this is intended for a GPU. Continuing on CPU.")

    session_start = time.monotonic()   # counts prep + train against the budget
    total_s = cfg["total_hours"] * 3600
    reserve_s = cfg["reserve_hours"] * 3600

    def remaining_train_s() -> float:
        return total_s - reserve_s - (time.monotonic() - session_start)

    summary = {
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "device": device, "seed_used": False,
        "family": family.to_dict(), "config": cfg,
        "shards": {}, "merges": {}, "eval_table": None,
    }

    def save_summary() -> None:
        with open(results_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # ---- Phase 1: prepare data -------------------------------------------
    prepare_if_needed(cfg, family, data_dir, log)
    save_summary()

    val_bin = data_dir / "val.bin"
    shard_names = [f"shard_{k + 1:02d}" for k in range(cfg["num_shards"])]
    shard_parts = [data_dir / f"part_{k + 1:02d}.bin" for k in range(cfg["num_shards"])]

    # ---- Phase 2: train shards (from scratch, time divided evenly) --------
    for i, (name, part) in enumerate(zip(shard_names, shard_parts)):
        out_dir = runs_dir / name
        final = out_dir / "final.pt"

        if final.exists():
            info = {"status": "skipped_done"}
            try:
                cm = load_checkpoint(final)["meta"]
                info.update({"steps": cm.get("steps"),
                             "epochs_completed": cm.get("epochs_completed"),
                             "tokens_seen": cm.get("tokens_seen"),
                             "elapsed_hours": (cm.get("elapsed_seconds") or 0) / 3600})
            except Exception:  # noqa: BLE001
                pass
            log(f"SKIP  {name}: already complete "
                f"(epochs={info.get('epochs_completed')}, tokens={info.get('tokens_seen')})")
            summary["shards"][name] = info
            save_summary()
            continue

        rem = remaining_train_s()
        # evenly divide the time left among the shards not yet trained this run
        n_left = sum(1 for n in shard_names[i:] if not (runs_dir / n / "final.pt").exists())
        if rem < cfg["min_shard_seconds"] or n_left == 0:
            log(f"DEADLINE: {rem / 60:.1f} min left; skipping {name} and going to merge/eval")
            summary["shards"][name] = {"status": "skipped_deadline"}
            save_summary()
            continue
        budget_h = (rem / n_left) / 3600

        log(f"TRAIN {name} on {part.name}: budget {budget_h:.2f}h "
            f"({n_left} shard(s) sharing {rem / 3600:.2f}h)")
        tcfg = TrainConfig(
            name=name, out_dir=str(out_dir),
            data_path=str(part), val_path=str(val_bin),
            family=family,
            n_heads_spec=cfg["n_heads"], d_ff_spec=cfg["d_ff"],
            init_from="scratch",                                  # no seed
            init_seed=cfg["shard_init_seed_base"] + i + 1,        # independent per shard
            batch_size=cfg["batch_size"], grad_accum=cfg["grad_accum"],
            seq_len=cfg["seq_len"], max_hours=budget_h,
            checkpoint_every_min=cfg["checkpoint_every_min"],
            log_interval=cfg["log_interval"],
            eval_interval=cfg["eval_interval"], eval_batches=cfg["eval_batches"],
            device="auto", dtype="auto", compile=cfg["compile"],
        )
        t0 = time.monotonic()
        try:
            s = run_training(tcfg)
            summary["shards"][name] = {
                "status": "done", "stop_reason": s["stop_reason"],
                "steps": s["steps"], "epochs_completed": s["epochs_completed"],
                "tokens_seen": s["tokens_seen"], "elapsed_hours": s["elapsed_hours"],
                "wall_hours_this_session": (time.monotonic() - t0) / 3600,
            }
            log(f"DONE  {name}: {s['stop_reason']}  steps={s['steps']}  "
                f"epochs={s['epochs_completed']}  tokens={s['tokens_seen']:,}  "
                f"elapsed={s['elapsed_hours']:.2f}h")
        except Exception as e:  # noqa: BLE001
            summary["shards"][name] = {"status": "failed", "error": str(e),
                                       "traceback": traceback.format_exc()}
            log(f"FAILED {name}: {e}  (continuing with remaining shards)")
        save_summary()

    # ---- Phase 3: merge ---------------------------------------------------
    shard_finals = [runs_dir / n / "final.pt" for n in shard_names]
    shard_finals = [p for p in shard_finals if p.exists()]
    log(f"MERGE phase: {len(shard_finals)} shard(s) available "
        f"({', '.join(p.parent.name for p in shard_finals)})")

    if len(shard_finals) >= 2:
        for m in cfg["merges"]:
            out = runs_dir / (m["name"] + ".pt")
            try:
                meta, _ = merge_checkpoints(
                    [str(p) for p in shard_finals], str(out),
                    alpha_mode=m.get("alpha_mode", "tokens"),
                    scaled=m.get("scaled", True), method=m.get("method", "stack"),
                    check=True, tol=m.get("tol", 1e-3), name=m["name"],
                )
                summary["merges"][m["name"]] = {
                    "status": "done", "method": meta["method"], "scaled": meta["scaled"],
                    "tokens_seen": meta["tokens_seen"],
                    "verify_max_err": meta.get("verify_max_err"),
                    "alphas": [e["alpha"] for e in meta["lineage"]],
                }
                err = meta.get("verify_max_err")
                log(f"MERGE {m['name']}: OK  "
                    + (f"exactness err={err:.2e}" if err is not None else "(baseline, no gate)"))
            except Exception as e:  # noqa: BLE001
                summary["merges"][m["name"]] = {"status": "failed", "error": str(e)}
                log(f"MERGE {m['name']}: FAILED {e}")
            save_summary()
    else:
        log("Not enough finished shards to merge (need >= 2).")

    # ---- Phase 4: evaluate -------------------------------------------------
    eval_models = list(shard_finals)
    for m in cfg["merges"]:
        mp = runs_dir / (m["name"] + ".pt")
        if mp.exists():
            eval_models.append(mp)
    eval_bins = [val_bin] + [
        data_dir / f"part_{k + 1:02d}.val.bin" for k in range(cfg["num_shards"])
        if (data_dir / f"part_{k + 1:02d}.val.bin").exists()
    ]

    if eval_models:
        log(f"EVAL phase: {len(eval_models)} model(s) x {len(eval_bins)} file(s)")
        try:
            res = evaluate_models(
                [str(p) for p in eval_models], [str(b) for b in eval_bins],
                seq_len=cfg["seq_len"], batch_size=cfg["batch_size"],
                max_batches=cfg["eval_max_batches"], device=device,
            )
            save_results(res, results_dir / "eval.json")
            table = format_results_table(res)
            summary["eval_table"] = table
            with open(results_dir / "eval_table.txt", "w", encoding="utf-8") as f:
                f.write(table + "\n")
            log("EVAL results:\n" + table)
        except Exception as e:  # noqa: BLE001
            log(f"EVAL FAILED: {e}")
            summary["eval_error"] = str(e)

    summary["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    summary["total_wall_hours"] = (time.monotonic() - session_start) / 3600
    save_summary()
    log(f"=== validation complete in {summary['total_wall_hours']:.2f}h ===")
    log(f"read: {results_dir / 'eval_table.txt'}  and  {results_dir / 'run_summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted — rerun the same command to resume where it stopped.")
        sys.exit(130)
