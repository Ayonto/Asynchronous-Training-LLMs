"""Shared training loop for seed models and shard models.

Asynchrony contract: this trainer knows nothing about other shards. It reads
one partition file, trains one model, and writes self-contained checkpoints.
Coordination between machines happens only through files.

Reliability contract (the "research PC crashes" requirement):
  * checkpoints are written atomically (temp file + rename) so a power cut
    mid-save can never corrupt the previous checkpoint;
  * `latest.pt` stores model, optimizer, step, token count, cumulative wall
    time, RNG states, and the data sampler's (epoch, cursor) position;
  * rerunning the SAME command auto-resumes from latest.pt and reproduces
    the exact batch sequence the uninterrupted run would have seen;
  * checkpoint cadence is per-run configurable (minutes and/or steps), so
    shard 1 can checkpoint hourly while shard 2 checkpoints every 3 hours.

Budget contract: training stops at whichever configured limit is hit first —
max_epochs (full passes over the partition), max_hours (cumulative wall time
across resumes), or max_steps (optimizer steps).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import signal
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from .config import FamilyConfig, ModelWidths
from .data import ChunkSampler, find_meta, get_batch, load_tokens, max_token_id
from .evaluate import evaluate_bin
from .model import (
    SAPModel,
    load_checkpoint,
    load_model,
    model_from_checkpoint,
    save_model_checkpoint,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # identity / io
    name: str                          # model name, e.g. "shard_03" — goes into meta
    out_dir: str                       # checkpoints + logs live here
    data_path: str                     # the partition .bin this model trains on
    val_path: Optional[str] = None     # optional .bin for periodic quick evals
    data_dtype: Optional[str] = None   # override if no meta.json next to the bin

    # family + widths (None + --init-from = inherit the checkpoint's widths)
    family: Optional[FamilyConfig] = None
    n_heads_spec: Optional[str] = None   # int or comma list per layer
    d_ff_spec: Optional[str] = None

    # initialization
    init_from: str = "scratch"         # 'scratch' | path to a seed/any checkpoint
    init_seed: int = 1337              # torch seed for weight init (per-shard!)
    fresh: bool = False                # ignore an existing latest.pt and restart

    # optimization
    batch_size: int = 16               # micro-batch (sequences per forward)
    grad_accum: int = 1                # optimizer batch = batch_size * grad_accum
    seq_len: Optional[int] = None      # default: family.max_seq_len
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    schedule: str = "cosine"           # 'cosine' | 'constant'
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    data_seed: int = 42                # epoch permutation seed

    # budget — whichever is hit first stops the run (all optional)
    max_epochs: Optional[int] = None
    max_hours: Optional[float] = None  # CUMULATIVE wall time across resumes
    max_steps: Optional[int] = None
    # "keep improving after the budget": relative budgets computed from the
    # checkpoint's own progress, so you never have to do cumulative arithmetic.
    # --extend-hours 6  == train 6 MORE hours on top of whatever is done;
    # --extend-epochs 1 == train 1 MORE full epoch. Work on fresh runs too
    # (then they equal max_hours/max_epochs from zero).
    extend_hours: Optional[float] = None
    extend_epochs: Optional[int] = None

    # checkpoint cadence (per-model, the user's requirement #5).
    # Progress-banking backoff: each process starts saving after
    # checkpoint_ramp_start_s, then DOUBLES the interval until it reaches
    # checkpoint_every_min. On an unstable machine this guarantees every
    # attempt banks progress before the typical crash window, while a stable
    # long run converges to the sparse steady cadence (few total writes).
    checkpoint_every_min: Optional[float] = 60.0
    checkpoint_every_steps: Optional[int] = None
    checkpoint_ramp_start_s: float = 60.0
    keep_prev: int = 3                 # previous latest.pt generations retained as
                                       # prev_1.pt (newest) .. prev_K.pt (oldest);
                                       # rotation is by cheap renames, never re-writes
    keep_history: int = 0              # extra step-stamped snapshots to retain

    # data transfer
    pin_memory: bool = True            # pinned-host-memory staging for GPU copies;
                                       # disable on machines with suspect RAM/driver

    # logging / eval
    log_interval: int = 20             # optimizer steps between log lines
    eval_interval: Optional[int] = 500 # optimizer steps between quick val evals
    eval_batches: int = 20

    # hardware
    device: str = "auto"               # 'auto' | 'cuda' | 'cpu'
    dtype: str = "auto"                # 'auto' | 'bf16' | 'fp32'
    compile: bool = False              # torch.compile the forward pass
    sdpa_backend: str = "auto"         # 'auto' | 'math': 'math' disables the fused
                                       # flash/mem-efficient attention kernels (slower,
                                       # same math) — use if training segfaults
    fused_adamw: bool = True           # fused AdamW is a native multi-tensor CUDA
                                       # kernel; disable on unstable driver stacks
                                       # (identical math, slightly slower step)

    def resolved_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# Changing these mid-run would silently corrupt the sampler / budget
# bookkeeping, so a resume refuses to proceed if they differ.
RESUME_CRITICAL_FIELDS = (
    "data_path", "batch_size", "grad_accum", "seq_len", "data_seed",
    "n_heads_spec", "d_ff_spec", "init_from",
)


def resolve_resume_checkpoint(out_dir: Path) -> Optional[Path]:
    """Find the newest usable checkpoint, tolerating a crash at ANY point of
    the save/rotate sequence:

      latest.pt        — the normal case
      latest.pt.new    — a fully-written new checkpoint that crashed between
                         rotation and its final rename (atomic writes mean a
                         partially-written file can never carry this name):
                         promote it to latest.pt and use it
      prev_1.pt ...    — rotation happened but the new latest was lost; fall
                         back to the newest previous generation

    Returns None when nothing usable exists (fresh start)."""
    latest = out_dir / "latest.pt"
    if latest.exists():
        return latest
    pending = out_dir / "latest.pt.new"
    if pending.exists():
        os.replace(pending, latest)
        print(f"[resume] promoted interrupted save {pending.name} -> latest.pt")
        return latest
    for prev in sorted(out_dir.glob("prev_*.pt")):
        print(f"[resume] latest.pt missing; falling back to {prev.name}")
        return prev
    return None


def _install_stop_handlers(stop_state: dict) -> list:
    """Install SIGINT/SIGTERM handlers that request a GRACEFUL stop: the loop
    finishes its current step, saves a checkpoint, writes status, and exits
    cleanly. Registration is defensive (Windows supports fewer signals; only
    the main thread may register). Returns (signum, old_handler) pairs so the
    caller can restore them."""
    installed = []
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            def _handler(signum, frame, _n=name):
                stop_state["signal"] = _n
            old = signal.signal(sig, _handler)
            installed.append((sig, old))
        except (ValueError, OSError, RuntimeError):
            pass   # non-main thread or unsupported on this platform
    return installed


# ---------------------------------------------------------------------------
# Model initialization: scratch, branch, or function-preserving grow
# ---------------------------------------------------------------------------

@torch.no_grad()
def grow_from_checkpoint(
    init_path: str,
    family: FamilyConfig,
    target_widths: Optional[ModelWidths],
    init_seed: int,
) -> Tuple[SAPModel, dict]:
    """Initialize a shard from a seed (or any family) checkpoint.

    Equal widths  -> plain branch: an exact copy.
    Wider target  -> function-preserving growth: the seed's heads/neurons are
    copied into the first slots; NEW units get random input-side weights but
    ZERO output-side weights (wo columns / w_down columns), so at branch time
    the grown model computes exactly the seed's function while the new units
    still receive gradient and differentiate during shard training.
    """
    src, src_family, src_widths, src_meta = load_model(init_path, device="cpu")
    if src_family.to_dict() != family.to_dict():
        raise ValueError(
            "init checkpoint belongs to a different family:\n"
            f"  checkpoint: {src_family.to_dict()}\n  requested:  {family.to_dict()}"
        )
    if target_widths is None:
        target_widths = src_widths

    for i in range(family.n_layers):
        if (target_widths.n_heads[i] < src_widths.n_heads[i]
                or target_widths.d_ff[i] < src_widths.d_ff[i]):
            raise ValueError(
                f"layer {i}: target widths (H={target_widths.n_heads[i]}, "
                f"d_ff={target_widths.d_ff[i]}) are smaller than the init "
                f"checkpoint's (H={src_widths.n_heads[i]}, d_ff={src_widths.d_ff[i]}); "
                "shrinking is not supported"
            )

    torch.manual_seed(init_seed)  # governs the random init of any NEW units
    model = SAPModel(family, target_widths)

    model.embed.weight.copy_(src.embed.weight)
    model.final_norm.weight.copy_(src.final_norm.weight)
    dh = family.d_head
    for i, (dst_b, src_b) in enumerate(zip(model.blocks, src.blocks)):
        dst_b.attn_norm.weight.copy_(src_b.attn_norm.weight)
        dst_b.ffn_norm.weight.copy_(src_b.ffn_norm.weight)

        hs = src_widths.n_heads[i] * dh
        dst_b.attn.wq.weight[:hs, :].copy_(src_b.attn.wq.weight)
        dst_b.attn.wk.weight[:hs, :].copy_(src_b.attn.wk.weight)
        dst_b.attn.wv.weight[:hs, :].copy_(src_b.attn.wv.weight)
        dst_b.attn.wo.weight[:, :hs].copy_(src_b.attn.wo.weight)
        dst_b.attn.wo.weight[:, hs:].zero_()      # new heads: silent at branch time

        fs = src_widths.d_ff[i]
        dst_b.ffn.w_gate.weight[:fs, :].copy_(src_b.ffn.w_gate.weight)
        dst_b.ffn.w_up.weight[:fs, :].copy_(src_b.ffn.w_up.weight)
        dst_b.ffn.w_down.weight[:, :fs].copy_(src_b.ffn.w_down.weight)
        dst_b.ffn.w_down.weight[:, fs:].zero_()   # new neurons: silent at branch time

    return model, src_meta


# ---------------------------------------------------------------------------
# Optimizer and LR schedule
# ---------------------------------------------------------------------------

def configure_optimizer(model: SAPModel, cfg: TrainConfig) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused_ok = cfg.resolved_device().type == "cuda" and cfg.fused_adamw
    return torch.optim.AdamW(
        groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=fused_ok
    )


def lr_at(step: int, cfg: TrainConfig, total_steps: Optional[int]) -> float:
    """Functional schedule — a pure function of the step count, so resumes
    never need scheduler state."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if cfg.schedule == "constant" or total_steps is None:
        return cfg.lr
    if step >= total_steps:
        return cfg.min_lr
    frac = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * frac))


# ---------------------------------------------------------------------------
# The training loop
# ---------------------------------------------------------------------------

def run_training(cfg: TrainConfig) -> dict:
    """Train one model to its budget. Returns a summary dict; writes
    out_dir/latest.pt (resumable) and out_dir/final.pt (portable, stripped)."""
    if cfg.family is None:
        raise ValueError("TrainConfig.family is required")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.resolved_device()

    if device.type == "cuda" and cfg.sdpa_backend == "math":
        # fall back to the plain math attention kernel: identical results,
        # no fused native kernels — the stable path on flaky driver stacks
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        print(f"[{cfg.name}] NOTE: fused attention kernels disabled (sdpa_backend=math)")

    # -- data ---------------------------------------------------------------
    tokens = load_tokens(cfg.data_path, dtype=cfg.data_dtype)
    meta_file = find_meta(cfg.data_path)
    if meta_file is not None and meta_file["vocab_size"] != cfg.family.vocab_size:
        raise ValueError(
            f"family vocab_size ({cfg.family.vocab_size}) != dataset vocab_size "
            f"({meta_file['vocab_size']}) — the tokenizer is part of the family "
            "skeleton; fix the family config"
        )
    val_tokens = load_tokens(cfg.val_path, dtype=cfg.data_dtype) if cfg.val_path else None
    seq_len = cfg.seq_len or cfg.family.max_seq_len
    if seq_len > cfg.family.max_seq_len:
        raise ValueError(f"seq_len {seq_len} > family max_seq_len {cfg.family.max_seq_len}")

    # -- data integrity gate --------------------------------------------------
    # The embedding table has vocab_size rows, but a uint16/uint32 .bin can
    # physically hold larger ids. A single out-of-range id (corrupt file, wrong
    # tokenizer, bit rot) would index OUT OF BOUNDS inside native CUDA code —
    # the one way bad data can hard-crash training. Scan the whole file once at
    # startup (seconds) and refuse with a readable error instead.
    for label, toks in (("train", tokens), ("val", val_tokens)):
        if toks is None:
            continue
        mx = max_token_id(toks)
        if mx >= cfg.family.vocab_size:
            raise ValueError(
                f"{label} data contains token id {mx} >= vocab_size "
                f"{cfg.family.vocab_size}. The .bin is corrupt or was tokenized "
                "with a different tokenizer than the family's. Re-run data "
                "preparation (and scripts/check_data.py) before training."
            )
        print(f"[{cfg.name}] data check ({label}): {len(toks):,} tokens, "
              f"max id {mx} < vocab {cfg.family.vocab_size} OK")

    # -- model: resume > init_from > scratch ---------------------------------
    latest_path = out_dir / "latest.pt"
    resume_path = None if cfg.fresh else resolve_resume_checkpoint(out_dir)
    resuming = resume_path is not None
    if (cfg.n_heads_spec is None) != (cfg.d_ff_spec is None):
        raise ValueError("give both --n-heads and --d-ff, or neither")
    widths = None
    if cfg.n_heads_spec is not None:
        widths = ModelWidths.from_spec(cfg.family.n_layers, cfg.n_heads_spec, cfg.d_ff_spec)
    seed_tokens = 0
    init_kind = "scratch"

    if resuming:
        ckpt = load_checkpoint(resume_path)
        saved_args = ckpt.get("train_args", {})
        for f in RESUME_CRITICAL_FIELDS:
            now = getattr(cfg, f)
            then = saved_args.get(f)
            if then is not None and str(then) != str(now):
                raise ValueError(
                    f"resume mismatch on '{f}': checkpoint has {then!r}, command has "
                    f"{now!r}. Changing this mid-run breaks the deterministic data/"
                    "budget bookkeeping. Use --fresh to restart deliberately."
                )
        model, _, widths, prev_meta = model_from_checkpoint(ckpt, device="cpu")
        seed_tokens = prev_meta.get("seed_tokens", 0)
        init_kind = prev_meta.get("init_from", "scratch")
    elif cfg.init_from != "scratch":
        # widths=None inherits the checkpoint's widths (plain branch);
        # explicit wider widths trigger function-preserving growth
        model, src_meta = grow_from_checkpoint(cfg.init_from, cfg.family, widths, cfg.init_seed)
        widths = model.widths
        seed_tokens = src_meta.get("tokens_seen", 0) or 0
        init_kind = cfg.init_from
        ckpt = None
    else:
        if widths is None:
            raise ValueError("training from scratch requires --n-heads and --d-ff")
        torch.manual_seed(cfg.init_seed)
        model = SAPModel(cfg.family, widths)
        ckpt = None

    model.to(device)
    raw_model = model                      # keep the uncompiled handle for saving
    if cfg.compile:
        model = torch.compile(model)

    optimizer = configure_optimizer(raw_model, cfg)

    # -- restore training state on resume ------------------------------------
    step = 0
    tokens_seen = 0
    elapsed_prev = 0.0
    sampler_state = {"epoch": 0, "cursor": 0}
    if resuming:
        ts = ckpt["train_state"]
        optimizer.load_state_dict(ckpt["optimizer_state"])
        step = ts["step"]
        tokens_seen = ts["tokens_seen"]
        elapsed_prev = ts["elapsed_seconds"]
        sampler_state = {"epoch": ts["epoch"], "cursor": ts["cursor"]}
        torch.set_rng_state(torch.tensor(ts["rng_torch"], dtype=torch.uint8))
        np.random.set_state(tuple(ts["rng_numpy"]))
        random.setstate(
            tuple(x if not isinstance(x, list) else tuple(x) for x in ts["rng_python"])
        )
        print(f"[{cfg.name}] resumed from {resume_path}: step {step}, "
              f"epoch {sampler_state['epoch']}, {tokens_seen:,} tokens, "
              f"{elapsed_prev / 3600:.2f}h elapsed")

    # -- relative budgets: "train N more on top of what's done" ---------------
    if cfg.extend_hours is not None:
        cfg.max_hours = elapsed_prev / 3600 + cfg.extend_hours
        print(f"[{cfg.name}] extend: +{cfg.extend_hours}h on top of "
              f"{elapsed_prev / 3600:.2f}h done -> total budget {cfg.max_hours:.2f}h")
    if cfg.extend_epochs is not None:
        cfg.max_epochs = sampler_state["epoch"] + cfg.extend_epochs
        print(f"[{cfg.name}] extend: +{cfg.extend_epochs} epoch(s) on top of "
              f"{sampler_state['epoch']} done -> total budget {cfg.max_epochs} epochs")

    sampler = ChunkSampler(
        n_tokens=len(tokens), seq_len=seq_len, batch_size=cfg.batch_size,
        data_seed=cfg.data_seed, epoch=sampler_state["epoch"],
        cursor=sampler_state["cursor"],
    )

    steps_per_epoch = max(1, sampler.batches_per_epoch // cfg.grad_accum)
    total_steps = cfg.max_steps
    if total_steps is None and cfg.max_epochs is not None:
        total_steps = steps_per_epoch * cfg.max_epochs
    if cfg.schedule == "cosine" and total_steps is None:
        print(f"[{cfg.name}] WARNING: cosine schedule needs a known horizon "
              "(max_steps or max_epochs); falling back to constant LR after warmup.")

    use_bf16 = (
        cfg.dtype == "bf16"
        or (cfg.dtype == "auto" and device.type == "cuda" and torch.cuda.is_bf16_supported())
    )
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_bf16 and device.type == "cuda" else nullcontext())

    counts = raw_model.param_counts()
    print(f"[{cfg.name}] {counts['total'] / 1e6:.1f}M params "
          f"({counts['stackable'] / 1e6:.1f}M stackable + "
          f"{counts['embedding'] / 1e6:.1f}M embedding) on {device}, "
          f"{'bf16' if use_bf16 else 'fp32'}; "
          f"{sampler.n_chunks:,} chunks/epoch, {steps_per_epoch:,} steps/epoch")

    log_path = out_dir / "train_log.jsonl"

    session_start = time.monotonic()

    def elapsed_total() -> float:
        return elapsed_prev + (time.monotonic() - session_start)

    def build_meta() -> dict:
        return {
            "role": "seed" if cfg.name.startswith("seed") else "shard",
            "name": cfg.name,
            "tokens_seen": tokens_seen,   # THIS shard's own tokens -> its alpha
            "seed_tokens": seed_tokens,   # informational; not counted into alpha
            "init_from": init_kind,
            "data_path": str(cfg.data_path),
            "steps": step,
            "epochs_completed": sampler.epoch,
            "elapsed_seconds": elapsed_total(),
        }

    def _write_checkpoint(path: Path, with_train_state: bool) -> None:
        train_state = None
        opt_state = None
        if with_train_state:
            train_state = {
                "step": step,
                "tokens_seen": tokens_seen,
                "epoch": sampler.epoch,
                "cursor": sampler.cursor,
                "elapsed_seconds": elapsed_total(),
                "rng_torch": torch.get_rng_state().tolist(),
                "rng_numpy": list(np.random.get_state()),
                "rng_python": list(random.getstate()),
            }
            opt_state = optimizer.state_dict()
        payload_args = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
                        if f.name != "family"}
        save_model_checkpoint(
            path, raw_model, build_meta(),
            optimizer_state=opt_state, train_state=train_state,
            extra={"train_args": payload_args},
        )

    def save(path: Path, with_train_state: bool = True) -> None:
        """Write a checkpoint. For latest.pt, retain the previous keep_prev
        generations as prev_1.pt (newest) .. prev_K.pt (oldest).

        Crash-safe ordering — at every instant at least one complete
        checkpoint exists and resolve_resume_checkpoint() can find it:
          1. serialize the NEW state atomically to latest.pt.new
          2. rotate: drop prev_K, shift prev_i -> prev_{i+1}, latest -> prev_1
          3. rename latest.pt.new -> latest.pt
        Rotation is pure renames — the big file is serialized exactly once."""
        if path != latest_path or cfg.keep_prev <= 0:
            _write_checkpoint(path, with_train_state)
            return
        pending = out_dir / "latest.pt.new"
        _write_checkpoint(pending, with_train_state)          # step 1 (atomic)
        oldest = out_dir / f"prev_{cfg.keep_prev}.pt"          # step 2
        if oldest.exists():
            oldest.unlink()
        for i in range(cfg.keep_prev - 1, 0, -1):
            p = out_dir / f"prev_{i}.pt"
            if p.exists():
                os.replace(p, out_dir / f"prev_{i + 1}.pt")
        if latest_path.exists():
            os.replace(latest_path, out_dir / "prev_1.pt")
        os.replace(pending, latest_path)                       # step 3

    def log(record: dict) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # -- lightweight live progress: out_dir/status.json ------------------------
    # One tiny atomic JSON per shard, refreshed every log_interval steps.
    # `scripts/progress.py` renders these for all shards in one table; nothing
    # extra is printed to this terminal.
    status_path = out_dir / "status.json"
    last_metrics = {"loss": None, "val_loss": None, "val_ppl": None}

    def write_status(state: str, error: Optional[str] = None,
                     stop_reason: Optional[str] = None) -> None:
        el_h = elapsed_total() / 3600
        progress = None
        if cfg.max_steps:
            progress = min(1.0, step / cfg.max_steps)
        elif cfg.max_epochs:
            done = sampler.epoch + sampler.cursor / max(1, sampler.batches_per_epoch)
            progress = min(1.0, done / cfg.max_epochs)
        elif cfg.max_hours:
            progress = min(1.0, el_h / cfg.max_hours)
        eta_h = (el_h * (1 - progress) / progress) if progress and progress > 0.001 else None
        payload = {
            "name": cfg.name,
            "state": state,                      # running | completed | stopped | failed
            "stop_reason": stop_reason,
            "error": error,
            "step": step,
            "epoch": sampler.epoch,
            "tokens_seen": tokens_seen,
            "loss": last_metrics["loss"],
            "val_loss": last_metrics["val_loss"],
            "val_ppl": last_metrics["val_ppl"],
            "elapsed_hours": round(el_h, 4),
            "budget": {"max_steps": cfg.max_steps, "max_epochs": cfg.max_epochs,
                       "max_hours": cfg.max_hours},
            "progress": round(progress, 4) if progress is not None else None,
            "eta_hours": round(eta_h, 3) if eta_h is not None else None,
            "pid": os.getpid(),
            "updated": time.time(),
        }
        tmp = status_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, status_path)

    # -- loop -----------------------------------------------------------------
    # Guaranteed-progress banking: every session (fresh start OR relaunch after
    # a crash) saves its first checkpoint after ~checkpoint_ramp_start_s, then
    # DOUBLES the interval each save until it reaches the steady cadence
    # (checkpoint_every_min). Net effect: a process that keeps crashing after
    # 10-15 minutes still banks several minutes of progress on EVERY attempt
    # (so the run provably advances), while a stable multi-hour run performs
    # only a handful of writes (1min, 2, 4, 8, ... -> cap).
    last_ckpt_time = time.monotonic()
    ckpt_interval_s = None
    ckpt_cap_s = None
    if cfg.checkpoint_every_min is not None:
        ckpt_cap_s = cfg.checkpoint_every_min * 60.0
        ckpt_interval_s = min(max(cfg.checkpoint_ramp_start_s, 1.0), ckpt_cap_s)
    stop_reason = None
    running_loss, running_n = 0.0, 0

    # graceful stop: Ctrl+C / kill <pid> finishes the current step, saves,
    # writes status, and exits cleanly — never a mid-write kill
    stop_state = {"signal": None}
    old_handlers = _install_stop_handlers(stop_state)

    write_status("running")
    model.train()
    try:
        while True:
            if stop_state["signal"] is not None:
                stop_reason = f"stopped_by_{stop_state['signal']}"
                break
            if cfg.max_steps is not None and step >= cfg.max_steps:
                stop_reason = "max_steps"
                break
            if cfg.max_epochs is not None and sampler.epoch >= cfg.max_epochs:
                stop_reason = "max_epochs"
                break
            if cfg.max_hours is not None and elapsed_total() >= cfg.max_hours * 3600:
                stop_reason = "max_hours"
                break

            lr = lr_at(step, cfg, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            step_loss = 0.0
            for _ in range(cfg.grad_accum):
                idxs = sampler.next_batch()
                x, y = get_batch(tokens, idxs, seq_len, device, pin=cfg.pin_memory)
                with amp:
                    logits, loss = model(x, targets=y)
                # free the (batch, seq, vocab) logits immediately — with a 50k
                # vocab this tensor is GB-scale and only the loss is needed
                del logits
                (loss / cfg.grad_accum).backward()
                step_loss += loss.item() / cfg.grad_accum
                tokens_seen += x.numel()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            running_loss += step_loss
            running_n += 1

            if step % cfg.log_interval == 0:
                avg = running_loss / running_n
                last_metrics["loss"] = avg
                rec = {"step": step, "loss": avg, "lr": lr, "epoch": sampler.epoch,
                       "tokens_seen": tokens_seen, "elapsed_h": elapsed_total() / 3600}
                print(f"[{cfg.name}] step {step:>7} | loss {avg:.4f} | lr {lr:.2e} | "
                      f"epoch {sampler.epoch} | {tokens_seen / 1e6:.1f}M tok | "
                      f"{rec['elapsed_h']:.2f}h")
                log(rec)
                write_status("running")
                running_loss, running_n = 0.0, 0

            if (cfg.eval_interval and val_tokens is not None
                    and step % cfg.eval_interval == 0):
                r = evaluate_bin(raw_model, val_tokens, seq_len=seq_len,
                                 batch_size=cfg.batch_size,
                                 max_batches=cfg.eval_batches, device=device)
                model.train()
                if device.type == "cuda":
                    # eval allocates differently-shaped tensors than training;
                    # release them so the allocator doesn't fragment
                    torch.cuda.empty_cache()
                last_metrics["val_loss"] = r["loss"]
                last_metrics["val_ppl"] = r["perplexity"]
                print(f"[{cfg.name}] step {step:>7} | VAL loss {r['loss']:.4f} | "
                      f"ppl {r['perplexity']:.2f}")
                log({"step": step, "val_loss": r["loss"], "val_ppl": r["perplexity"]})
                write_status("running")

            due_time = (ckpt_interval_s is not None
                        and time.monotonic() - last_ckpt_time >= ckpt_interval_s)
            due_steps = (cfg.checkpoint_every_steps is not None
                         and step % cfg.checkpoint_every_steps == 0)
            if due_time or due_steps:
                save(latest_path)
                last_ckpt_time = time.monotonic()
                if ckpt_interval_s is not None:
                    ckpt_interval_s = min(ckpt_interval_s * 2.0, ckpt_cap_s)
                print(f"[{cfg.name}] checkpoint saved at step {step} -> {latest_path}")
                if cfg.keep_history > 0:
                    snap = out_dir / f"ckpt_step{step:08d}.pt"
                    save(snap)
                    snaps = sorted(out_dir.glob("ckpt_step*.pt"))
                    for old in snaps[: max(0, len(snaps) - cfg.keep_history)]:
                        old.unlink()

    except Exception as e:
        # GRACEFUL FAILURE: bank whatever progress exists, record the reason in
        # a place the progress viewer can show, print it clearly, then re-raise
        # so the process exits nonzero (the orchestrator sees a real failure).
        reason = f"{type(e).__name__}: {e}"
        if isinstance(e, torch.cuda.OutOfMemoryError):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()   # make room so the emergency save works
            reason += "  (fix: lower --batch-size and raise --grad-accum)"
        try:
            save(latest_path)
            saved_note = f"progress saved to {latest_path}"
        except Exception as save_err:  # noqa: BLE001
            saved_note = f"could not save a final checkpoint ({save_err}); " \
                         "the last periodic checkpoint is still on disk"
        write_status("failed", error=reason)
        print(f"\n[{cfg.name}] TRAINING FAILED\n"
              f"  reason : {reason}\n"
              f"  state  : {saved_note}\n"
              f"  resume : rerun the same command — it continues from the last checkpoint\n",
              flush=True)
        log({"event": "failed", "error": reason, "step": step,
             "tokens_seen": tokens_seen})
        raise
    finally:
        for sig, old in old_handlers:
            try:
                signal.signal(sig, old)
            except (ValueError, OSError, RuntimeError):
                pass

    # -- end of run: budget reached, or a graceful stop request ----------------
    save(latest_path)                                   # resumable
    completed = stop_reason in ("max_steps", "max_epochs", "max_hours")
    final_path = out_dir / "final.pt"
    if completed:
        # final.pt (no optimizer state) marks "budget completed" — the portable
        # artifact for merge/eval. A signal-stop does NOT write it: the shard is
        # paused, not finished; rerun (optionally with --extend-*) to continue.
        save(final_path, with_train_state=False)
    summary = {
        "stop_reason": stop_reason,
        "steps": step,
        "epochs_completed": sampler.epoch,
        "tokens_seen": tokens_seen,
        "elapsed_hours": elapsed_total() / 3600,
        "final_checkpoint": str(final_path) if completed else None,
        "latest_checkpoint": str(latest_path),
    }
    write_status("completed" if completed else "stopped", stop_reason=stop_reason)
    print(f"[{cfg.name}] {'done' if completed else 'stopped'} ({stop_reason}): "
          f"{step} steps, {sampler.epoch} epochs, {tokens_seen:,} tokens, "
          f"{summary['elapsed_hours']:.2f}h"
          + (f" -> {final_path}" if completed else
             " — paused; rerun (or --extend-hours/--extend-epochs) to continue"))
    log({"event": "finished" if completed else "stopped", **summary})
    return summary
