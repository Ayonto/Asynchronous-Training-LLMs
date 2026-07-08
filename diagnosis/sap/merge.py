"""The Stack-and-Scale merge operator — the core contribution of the thesis.

For N parent models with output scales s_1..s_N, per transformer block:

  attention   wq/wk/wv : concatenate the parents' head blocks (UNSCALED)
              wo       : concatenate the parents' head blocks scaled by s_k
  FFN         w_gate/w_up : concatenate the parents' neuron rows (UNSCALED)
              w_down      : concatenate the parents' neuron columns scaled by s_k
  norms       each parent's RMSNorm gain is absorbed (exactly) into its own
              input-side projections first; merged gains are all-ones
  embeddings  weighted average (the single parameter-level step; they cannot
              stack because vocab x d_model is fixed by the skeleton)

Scaled mode:   s_k = alpha_k with alpha >= 0, sum(alpha) = 1
               -> merged sublayer = SUM_k alpha_k * parent_k sublayer   (exact average)
Unscaled mode: s_k = 1
               -> merged sublayer = SUM_k parent_k sublayer             (exact sum)

Both identities are enforced at merge time by `verify_merge` (block-by-block,
float64) — a merge that does not satisfy its own math refuses to save.
"""

from __future__ import annotations

import copy
import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch

from .config import FamilyConfig, ModelWidths, merged_widths
from .model import (
    SAPModel,
    build_rope_cache,
    load_checkpoint,
    model_from_checkpoint,
    save_model_checkpoint,
)


class MergeExactnessError(AssertionError):
    """Raised when a merged model fails its own exactness gate."""


# ---------------------------------------------------------------------------
# Norm-gain absorption
# ---------------------------------------------------------------------------

@torch.no_grad()
def absorb_norm_gains(model: SAPModel) -> SAPModel:
    """Fold each block's RMSNorm gains into the following projections, in place.

    RMSNorm(x) = g * (x / rms(x)). Because g is diagonal and rms() has no
    parameters, W @ (g * u) == (W * g) @ u exactly — so multiplying the
    input columns of wq/wk/wv by the attention-norm gain (and w_gate/w_up by
    the FFN-norm gain) and setting the gains to 1 leaves the model's
    function unchanged. After absorption every head and neuron receives
    exactly the normalized input it was trained on, so gains never need to
    be averaged across parents.

    The FINAL pre-LM-head norm cannot be absorbed (the LM head is shared and
    averaged), so it is left alone here and averaged in the merge — the one
    known, small approximation besides the embedding average.
    """
    for blk in model.blocks:
        g_attn = blk.attn_norm.weight
        for lin in (blk.attn.wq, blk.attn.wk, blk.attn.wv):
            lin.weight.mul_(g_attn)          # weight (out, in) * g (in,) -> scale input dims
        blk.attn_norm.weight.fill_(1.0)
        g_ffn = blk.ffn_norm.weight
        for lin in (blk.ffn.w_gate, blk.ffn.w_up):
            lin.weight.mul_(g_ffn)
        blk.ffn_norm.weight.fill_(1.0)
    return model


# ---------------------------------------------------------------------------
# Mixing coefficients
# ---------------------------------------------------------------------------

def resolve_alphas(
    n_models: int,
    alpha_mode: str = "tokens",
    manual_alphas: Optional[Sequence[float]] = None,
    token_counts: Optional[Sequence[int]] = None,
) -> List[float]:
    """Return convex weights alpha_1..alpha_N (>= 0, summing to 1).

      tokens  — alpha_k = tokens_k / sum(tokens)   (the thesis default)
      uniform — alpha_k = 1/N
      manual  — user-supplied, normalized to sum to 1
    """
    if alpha_mode == "manual":
        if manual_alphas is None or len(manual_alphas) != n_models:
            raise ValueError(f"manual alpha mode needs exactly {n_models} values")
        if any(a < 0 for a in manual_alphas):
            raise ValueError("alphas must be non-negative")
        total = float(sum(manual_alphas))
        if total <= 0:
            raise ValueError("alphas sum to zero")
        return [float(a) / total for a in manual_alphas]
    if alpha_mode == "uniform":
        return [1.0 / n_models] * n_models
    if alpha_mode == "tokens":
        if token_counts is None or any(t is None for t in token_counts):
            raise ValueError("token-weighted alphas need tokens_seen in every checkpoint")
        if any(t <= 0 for t in token_counts):
            raise ValueError(
                f"token counts {list(token_counts)} contain zero — a model that saw no "
                "tokens has no data-proportional weight; use --alpha-mode uniform or manual"
            )
        total = float(sum(token_counts))
        return [float(t) / total for t in token_counts]
    raise ValueError(f"unknown alpha mode: {alpha_mode}")


# ---------------------------------------------------------------------------
# The merge operator
# ---------------------------------------------------------------------------

def _check_same_family(models: Sequence[SAPModel]) -> FamilyConfig:
    family = models[0].family
    for m in models[1:]:
        if m.family.to_dict() != family.to_dict():
            raise ValueError(
                "models are not members of the same family:\n"
                f"  {family.to_dict()}\nvs\n  {m.family.to_dict()}"
            )
    return family


@torch.no_grad()
def merge_models(
    models: Sequence[SAPModel],
    alphas: Sequence[float],
    scaled: bool = True,
) -> SAPModel:
    """Stack-and-Scale N-way merge. Does NOT mutate the parents.

    `alphas` are always required and always used for the embedding / final
    norm average (those must stay a convex combination regardless of mode).
    `scaled` controls the output-side scaling only:
      scaled=True  -> wo / w_down slices multiplied by alpha_k (function average)
      scaled=False -> wo / w_down slices unscaled                (function sum)
    """
    if len(models) < 2:
        raise ValueError("need at least two models to merge")
    if len(alphas) != len(models):
        raise ValueError("one alpha per model required")
    if abs(sum(alphas) - 1.0) > 1e-8:
        raise ValueError(f"alphas must sum to 1, got {sum(alphas)}")
    family = _check_same_family(models)

    out_scales = list(alphas) if scaled else [1.0] * len(models)
    widths = merged_widths([m.widths for m in models])
    dtype = next(models[0].parameters()).dtype

    merged = SAPModel(family, widths)
    merged.to(dtype)

    F64 = torch.float64  # all surgery arithmetic in float64, cast once at the end

    def f64(t: torch.Tensor) -> torch.Tensor:
        return t.detach().to("cpu", torch.float64)

    for li, mblk in enumerate(merged.blocks):
        src = [m.blocks[li] for m in models]

        # --- attention: absorb each parent's attn-norm gain, then stack heads
        g = [f64(b.attn_norm.weight) for b in src]
        wq = torch.cat([f64(b.attn.wq.weight) * g[k] for k, b in enumerate(src)], dim=0)
        wk = torch.cat([f64(b.attn.wk.weight) * g[k] for k, b in enumerate(src)], dim=0)
        wv = torch.cat([f64(b.attn.wv.weight) * g[k] for k, b in enumerate(src)], dim=0)
        wo = torch.cat([out_scales[k] * f64(b.attn.wo.weight) for k, b in enumerate(src)], dim=1)
        mblk.attn.wq.weight.copy_(wq.to(dtype))
        mblk.attn.wk.weight.copy_(wk.to(dtype))
        mblk.attn.wv.weight.copy_(wv.to(dtype))
        mblk.attn.wo.weight.copy_(wo.to(dtype))
        mblk.attn_norm.weight.fill_(1.0)

        # --- FFN: absorb each parent's ffn-norm gain, then stack neurons
        h = [f64(b.ffn_norm.weight) for b in src]
        wg = torch.cat([f64(b.ffn.w_gate.weight) * h[k] for k, b in enumerate(src)], dim=0)
        wu = torch.cat([f64(b.ffn.w_up.weight) * h[k] for k, b in enumerate(src)], dim=0)
        wd = torch.cat([out_scales[k] * f64(b.ffn.w_down.weight) for k, b in enumerate(src)], dim=1)
        mblk.ffn.w_gate.weight.copy_(wg.to(dtype))
        mblk.ffn.w_up.weight.copy_(wu.to(dtype))
        mblk.ffn.w_down.weight.copy_(wd.to(dtype))
        mblk.ffn_norm.weight.fill_(1.0)

    # --- embeddings + final norm: alpha-weighted average (parameter-level step)
    emb = sum(a * f64(m.embed.weight) for a, m in zip(alphas, models))
    fng = sum(a * f64(m.final_norm.weight) for a, m in zip(alphas, models))
    merged.embed.weight.copy_(emb.to(dtype))
    merged.final_norm.weight.copy_(fng.to(dtype))

    return merged


# ---------------------------------------------------------------------------
# Baseline: naive weight averaging (for the graceful-vs-catastrophic experiment)
# ---------------------------------------------------------------------------

@torch.no_grad()
def weight_average(models: Sequence[SAPModel], alphas: Sequence[float]) -> SAPModel:
    """Parameter-level weighted average. Only defined when every parent has
    identical widths. This is the baseline the thesis predicts collapses as
    shards diverge (permutation symmetry: unit i of A is summed with the
    unrelated unit i of B)."""
    if abs(sum(alphas) - 1.0) > 1e-8:
        raise ValueError("alphas must sum to 1")
    family = _check_same_family(models)
    w0 = models[0].widths.to_dict()
    for m in models[1:]:
        if m.widths.to_dict() != w0:
            raise ValueError(
                "weight averaging requires identical widths in every parent; "
                "use the stack merge for heterogeneous models"
            )
    dtype = next(models[0].parameters()).dtype
    avg = SAPModel(family, models[0].widths)
    avg.to(dtype)
    out_state = avg.state_dict()
    states = [m.state_dict() for m in models]
    for key in out_state:
        acc = sum(a * s[key].to(torch.float64) for a, s in zip(alphas, states))
        out_state[key] = acc.to(dtype)
    avg.load_state_dict(out_state)
    return avg


# ---------------------------------------------------------------------------
# The exactness gate — every merge must pass this before it is saved
# ---------------------------------------------------------------------------

@torch.no_grad()
def verify_merge(
    parents: Sequence[SAPModel],
    merged: SAPModel,
    alphas: Sequence[float],
    scaled: bool = True,
    n_tokens: int = 8,
    batch: int = 2,
    tol: float = 1e-3,
    generator_seed: int = 0,
) -> dict:
    """Verify, block by block in float64, that the merged model satisfies the
    claimed identity against the ORIGINAL (un-absorbed) parents:

      scaled:   merged_sublayer(x) == sum_k alpha_k * parent_k_sublayer(x)
      unscaled: merged_sublayer(x) == sum_k parent_k_sublayer(x)

    Comparing against un-absorbed parents means the gate also covers the
    norm-absorption step, not just the concatenation.

    Runs on CPU with tiny inputs, one block at a time (deep copies), so it is
    cheap even for large models. Raises MergeExactnessError on failure.

    Note on tolerance: for float64 models the identity holds to ~1e-13. For
    float32 models the merged weights carry one float32 rounding of the
    absorb/scale arithmetic (~1e-7 relative), so observed errors are ~1e-5;
    the default tol of 1e-3 is far above rounding noise and far below any
    real merge bug (a wrong axis, scale, or slice produces errors of order 1).
    """
    family = _check_same_family(list(parents) + [merged])
    out_scales = list(alphas) if scaled else [1.0] * len(parents)

    # widths bookkeeping must be the per-layer sums
    expect = merged_widths([p.widths for p in parents]).to_dict()
    if merged.widths.to_dict() != expect:
        raise MergeExactnessError(
            f"merged widths {merged.widths.to_dict()} != expected sums {expect}"
        )

    gen = torch.Generator().manual_seed(generator_seed)
    x = torch.randn(batch, n_tokens, family.d_model, dtype=torch.float64, generator=gen)
    cos, sin = build_rope_cache(n_tokens, family.d_head, family.rope_theta)  # float64

    per_layer = []
    for li in range(family.n_layers):
        mblk = copy.deepcopy(merged.blocks[li]).to("cpu").double()
        pblks = [copy.deepcopy(p.blocks[li]).to("cpu").double() for p in parents]

        t_attn = sum(s * b.attn(b.attn_norm(x), cos, sin) for s, b in zip(out_scales, pblks))
        g_attn = mblk.attn(mblk.attn_norm(x), cos, sin)
        e_attn = (g_attn - t_attn).abs().max().item()

        t_ffn = sum(s * b.ffn(b.ffn_norm(x)) for s, b in zip(out_scales, pblks))
        g_ffn = mblk.ffn(mblk.ffn_norm(x))
        e_ffn = (g_ffn - t_ffn).abs().max().item()

        per_layer.append({"layer": li, "attn_err": e_attn, "ffn_err": e_ffn})

    # embedding / final norm: definitionally an alpha-average — verify the copy
    emb_t = sum(a * p.embed.weight.detach().to(torch.float64) for a, p in zip(alphas, parents))
    e_emb = (merged.embed.weight.detach().to(torch.float64) - emb_t).abs().max().item()
    fng_t = sum(a * p.final_norm.weight.detach().to(torch.float64) for a, p in zip(alphas, parents))
    e_fng = (merged.final_norm.weight.detach().to(torch.float64) - fng_t).abs().max().item()

    max_err = max(
        max(r["attn_err"] for r in per_layer),
        max(r["ffn_err"] for r in per_layer),
        e_emb,
        e_fng,
    )
    report = {
        "max_err": max_err,
        "embed_err": e_emb,
        "final_norm_err": e_fng,
        "per_layer": per_layer,
        "scaled": scaled,
        "tol": tol,
    }
    if max_err > tol:
        raise MergeExactnessError(
            f"merge exactness gate FAILED: max error {max_err:.3e} > tol {tol:.1e}. "
            "The merged model does not compute the claimed combination of its "
            "parents — refusing to proceed."
        )
    return report


# ---------------------------------------------------------------------------
# High-level entry point used by scripts/merge.py
# ---------------------------------------------------------------------------

def merge_checkpoints(
    checkpoint_paths: Sequence[Union[str, Path]],
    out_path: Union[str, Path],
    alpha_mode: str = "tokens",
    manual_alphas: Optional[Sequence[float]] = None,
    scaled: bool = True,
    method: str = "stack",
    check: bool = True,
    tol: float = 1e-3,
    name: Optional[str] = None,
) -> Tuple[dict, Optional[dict]]:
    """Load N checkpoints, merge, verify, save. Returns (meta, verify_report).

    method 'stack' — the SAP structural merge (the thesis method)
    method 'avg'   — naive weight-average baseline (identical widths only)
    """
    ckpts = [load_checkpoint(p) for p in checkpoint_paths]
    loaded = [model_from_checkpoint(c, device="cpu") for c in ckpts]
    models = [t[0] for t in loaded]
    metas = [t[3] for t in loaded]

    token_counts = [m.get("tokens_seen") for m in metas]
    alphas = resolve_alphas(
        len(models), alpha_mode=alpha_mode,
        manual_alphas=manual_alphas, token_counts=token_counts,
    )

    if method == "stack":
        merged = merge_models(models, alphas, scaled=scaled)
        report = verify_merge(models, merged, alphas, scaled=scaled, tol=tol) if check else None
    elif method == "avg":
        merged = weight_average(models, alphas)
        report = None  # no exactness identity exists for weight averaging
    else:
        raise ValueError(f"unknown merge method: {method}")

    total_tokens = sum(t or 0 for t in token_counts)
    meta = {
        "role": "merged" if method == "stack" else "baseline_avg",
        "name": name or Path(out_path).stem,
        "method": method,
        "scaled": scaled,
        "alpha_mode": alpha_mode,
        "tokens_seen": total_tokens,   # continual-merge bookkeeping (report section 7.3)
        "lineage": [
            {
                "name": m.get("name", str(p)),
                "role": m.get("role", "unknown"),
                "tokens_seen": t,
                "alpha": a,
                "path": str(p),
            }
            for p, m, t, a in zip(checkpoint_paths, metas, token_counts, alphas)
        ],
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "verify_max_err": report["max_err"] if report else None,
    }
    save_model_checkpoint(out_path, merged, meta)
    return meta, report
