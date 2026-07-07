"""Evaluation: cross-entropy / perplexity / next-token accuracy on any .bin.

Deterministic by construction: evaluation walks the file in sequential,
non-overlapping windows (no shuffling), so two runs on the same file and
model produce identical numbers (up to GPU non-determinism in bf16; pass
fp32=True for bit-stable numbers).
"""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from .data import load_tokens
from .model import SAPModel, load_model


@torch.no_grad()
def evaluate_bin(
    model: SAPModel,
    tokens: np.memmap,
    seq_len: Optional[int] = None,
    batch_size: int = 16,
    max_batches: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    fp32: bool = False,
) -> Dict[str, float]:
    """Average next-token cross-entropy over sequential windows of the file.

    Returns {loss, perplexity, top1_accuracy, tokens_evaluated, batches}.
    Loss is token-weighted (every batch has the same token count, so the
    plain mean over batches is exact).
    """
    model.eval()
    device = torch.device(device)
    seq_len = seq_len or model.family.max_seq_len
    seq_len = min(seq_len, model.family.max_seq_len)

    n_chunks = (len(tokens) - 1) // seq_len
    if n_chunks < 1:
        raise ValueError(f"file too small for seq_len={seq_len} ({len(tokens)} tokens)")
    n_batches = max(1, n_chunks // batch_size)
    if max_batches is not None:
        n_batches = min(n_batches, max_batches)

    use_amp = device.type == "cuda" and not fp32
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    for b in range(n_batches):
        lo = b * batch_size
        idxs = np.arange(lo, min(lo + batch_size, n_chunks))
        xs = np.stack([tokens[i * seq_len: i * seq_len + seq_len] for i in idxs]).astype(np.int64)
        ys = np.stack([tokens[i * seq_len + 1: i * seq_len + seq_len + 1] for i in idxs]).astype(np.int64)
        x = torch.from_numpy(xs).to(device)
        y = torch.from_numpy(ys).to(device)
        with amp:
            logits, loss = model(x, targets=y)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    mean_loss = total_loss / total_tokens
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 50.0)),  # clamp: exp overflow guard
        "top1_accuracy": total_correct / total_tokens,
        "tokens_evaluated": total_tokens,
        "batches": n_batches,
    }


def evaluate_models(
    model_paths: Sequence[Union[str, Path]],
    bin_paths: Sequence[Union[str, Path]],
    seq_len: Optional[int] = None,
    batch_size: int = 16,
    max_batches: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    fp32: bool = False,
    dtype: Optional[str] = None,
) -> dict:
    """Evaluate every model on every .bin file. The dynamic test harness:
    hand it any set of checkpoints (shards, seed, merged, baseline) and any
    set of token files (val.bin, per-partition vals, ...) and it produces the
    full results grid."""
    bins = {str(p): load_tokens(p, dtype=dtype) for p in bin_paths}
    results = {}
    for mp in model_paths:
        model, family, widths, meta = load_model(mp, device=device)
        counts = model.param_counts()
        row = {
            "meta": {
                "role": meta.get("role", "unknown"),
                "name": meta.get("name", Path(mp).stem),
                "tokens_seen": meta.get("tokens_seen"),
                "params_total": counts["total"],
                "params_stackable": counts["stackable"],
                "n_heads": widths.n_heads,
                "d_ff": widths.d_ff,
            },
            "data": {},
        }
        for bp, toks in bins.items():
            row["data"][bp] = evaluate_bin(
                model, toks, seq_len=seq_len, batch_size=batch_size,
                max_batches=max_batches, device=device, fp32=fp32,
            )
        results[str(mp)] = row
        del model
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    return results


def format_results_table(results: dict) -> str:
    """Human-readable grid: one row per model, one column group per data file."""
    bin_names = []
    for row in results.values():
        for bp in row["data"]:
            if bp not in bin_names:
                bin_names.append(bp)
    short = {bp: Path(bp).name for bp in bin_names}

    lines = []
    header = f"{'model':<28} {'params':>10} {'tokens_seen':>13}"
    for bp in bin_names:
        header += f" | {short[bp]:>12} {'ppl':>9} {'acc':>6}"
    lines.append(header)
    lines.append("-" * len(header))
    for mp, row in results.items():
        m = row["meta"]
        tok = m["tokens_seen"]
        line = (
            f"{(m['name'] or Path(mp).stem)[:28]:<28} "
            f"{m['params_total'] / 1e6:>9.1f}M "
            f"{(f'{tok:,}' if tok else '-'):>13}"
        )
        for bp in bin_names:
            r = row["data"].get(bp)
            if r is None:
                line += f" | {'-':>12} {'-':>9} {'-':>6}"
            else:
                line += f" | {r['loss']:>12.4f} {r['perplexity']:>9.2f} {r['top1_accuracy']:>6.3f}"
        lines.append(line)
    return "\n".join(lines)


def save_results(results: dict, path: Union[str, Path]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


@torch.no_grad()
def generate_text(
    model_path: Union[str, Path],
    tokenizer_name: str,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    device: Union[str, torch.device] = "cpu",
) -> str:
    """Qualitative sample from any checkpoint. Requires the family tokenizer."""
    from transformers import AutoTokenizer  # lazy
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model, family, _, _ = load_model(model_path, device=device)
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("prompt tokenized to nothing")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())
