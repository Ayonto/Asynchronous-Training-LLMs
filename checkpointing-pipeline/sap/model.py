"""The family transformer: decoder-only, RMSNorm (pre-norm), RoPE, SwiGLU,
tied embeddings, causal attention.

Everything is implemented from scratch in plain PyTorch — no prebuilt
transformer modules — because the merge operator does tensor surgery on
these exact matrices and their layout must be fully under our control.

Merge-critical layout facts (nn.Linear stores weight as (out_features, in_features)):

  attn.wq / wk / wv : weight (H*d_head, d_model)   -> heads live on dim 0 (rows)
  attn.wo           : weight (d_model, H*d_head)   -> heads live on dim 1 (columns)
  ffn.w_gate / w_up : weight (d_ff, d_model)       -> neurons live on dim 0 (rows)
  ffn.w_down        : weight (d_model, d_ff)       -> neurons live on dim 1 (columns)

The number of heads is NEVER assumed to satisfy H*d_head == d_model.
Merged models are over-complete (H*d_head > d_model) and wo is rectangular.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FamilyConfig, ModelWidths

CHECKPOINT_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """y = g * x / rms(x).  The gain g is diagonal and rms() is parameter-free,
    which is exactly what makes gain absorption in the merge exact."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def build_rope_cache(max_seq_len: int, d_head: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin tables in float64 (cast down at use time).

    Returns cos, sin of shape (max_seq_len, d_head // 2).
    Half-split (Llama-style) rotation convention; must be identical for
    every family member, which it is because theta and d_head are skeleton.
    """
    if d_head % 2 != 0:
        raise ValueError("d_head must be even for RoPE")
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_head, 2, dtype=torch.float64) / d_head))
    t = torch.arange(max_seq_len, dtype=torch.float64)
    freqs = torch.outer(t, inv_freq)                       # (T, d_head/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate q or k. x: (B, H, T, d_head); cos/sin: (T, d_head/2)."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    """Causal multi-head attention with per-head RoPE.

    Each head is fully self-contained: its own Q/K/V slices, its own softmax,
    its own rows... columns of wq/wk/wv and its own row-block of wo. That
    self-containment is the entire basis of head-stacking merges.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.wq = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.wk = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.wv = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.wo = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.d_head
        q = self.wq(x).view(B, T, H, Dh).transpose(1, 2)   # (B, H, T, Dh)
        k = self.wk(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.wv(x).view(B, T, H, Dh).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, H * Dh)
        return self.wo(y)


class SwiGLU(nn.Module):
    """FFN(x) = w_down( SiLU(w_gate x) * (w_up x) ).

    Each hidden neuron owns one row of w_gate, one row of w_up and one
    column of w_down — neurons never interact inside the layer, so the
    layer output is a sum of per-neuron contributions (mergeable)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_ff = d_ff
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    """Pre-norm transformer block: x += attn(norm(x)); x += ffn(norm(x))."""

    def __init__(self, family: FamilyConfig, n_heads: int, d_ff: int):
        super().__init__()
        self.attn_norm = RMSNorm(family.d_model, family.norm_eps)
        self.attn = Attention(family.d_model, n_heads, family.d_head)
        self.ffn_norm = RMSNorm(family.d_model, family.norm_eps)
        self.ffn = SwiGLU(family.d_model, d_ff)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class SAPModel(nn.Module):
    """Decoder-only LM: embed -> L blocks -> final RMSNorm -> tied LM head."""

    def __init__(self, family: FamilyConfig, widths: ModelWidths):
        super().__init__()
        if widths.n_layers != family.n_layers:
            raise ValueError(
                f"widths describe {widths.n_layers} layers but family has {family.n_layers}"
            )
        self.family = family
        self.widths = widths

        self.embed = nn.Embedding(family.vocab_size, family.d_model)
        self.blocks = nn.ModuleList(
            Block(family, widths.n_heads[i], widths.d_ff[i]) for i in range(family.n_layers)
        )
        self.final_norm = RMSNorm(family.d_model, family.norm_eps)
        # LM head is TIED to the embedding (skeleton decision): logits = x @ embed.weight.T

        cos, sin = build_rope_cache(family.max_seq_len, family.d_head, family.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2-style scaled init for residual-output projections
        resid_std = 0.02 / math.sqrt(2 * family.n_layers)
        for blk in self.blocks:
            nn.init.normal_(blk.attn.wo.weight, mean=0.0, std=resid_std)
            nn.init.normal_(blk.ffn.w_down.weight, mean=0.0, std=resid_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -- forward --------------------------------------------------------------

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        if T > self.family.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.family.max_seq_len}")
        x = self.embed(idx)
        cos = self.rope_cos[:T].to(x.dtype)
        sin = self.rope_sin[:T].to(x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.final_norm(x)
        logits = F.linear(x, self.embed.weight)           # tied LM head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Simple sampling loop for qualitative checks."""
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.family.max_seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    # -- accounting -----------------------------------------------------------

    def param_counts(self) -> dict:
        """Split parameters into E (embedding, shared across merges) and
        P = S - E (stackable). This is the growth law S_merged = E + sum(S_i - E)."""
        total = sum(p.numel() for p in self.parameters())
        E = self.embed.weight.numel()
        return {"total": total, "embedding": E, "stackable": total - E}


# ---------------------------------------------------------------------------
# Checkpoint I/O — one canonical schema for seed / shard / merged models
# ---------------------------------------------------------------------------

def widths_from_state_dict(state: dict, family: FamilyConfig) -> ModelWidths:
    """Derive per-layer (n_heads, d_ff) from weight shapes.

    This is the report's cardinal implementation rule: head count is read
    from the tensors, never assumed from a constant, so merged (over-complete)
    models load cleanly."""
    n_heads, d_ff = [], []
    for i in range(family.n_layers):
        wq_out = state[f"blocks.{i}.attn.wq.weight"].shape[0]
        if wq_out % family.d_head != 0:
            raise ValueError(f"layer {i}: wq rows ({wq_out}) not divisible by d_head")
        n_heads.append(wq_out // family.d_head)
        d_ff.append(state[f"blocks.{i}.ffn.w_gate.weight"].shape[0])
    return ModelWidths(n_heads=n_heads, d_ff=d_ff)


def atomic_torch_save(payload: dict, path: Union[str, Path]) -> None:
    """Write to a temp file in the same directory, then rename. A crash or
    power loss mid-save can never corrupt an existing checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_model_checkpoint(
    path: Union[str, Path],
    model: SAPModel,
    meta: dict,
    optimizer_state: Optional[dict] = None,
    train_state: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> None:
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "family": model.family.to_dict(),
        "widths": model.widths.to_dict(),
        "model_state": model.state_dict(),
        "meta": meta,
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    if train_state is not None:
        payload["train_state"] = train_state
    if extra:
        payload.update(extra)
    atomic_torch_save(payload, path)


def load_checkpoint(path: Union[str, Path]) -> dict:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if "format_version" not in ckpt:
        raise ValueError(f"{path} is not a SAP checkpoint (missing format_version)")
    return ckpt


def model_from_checkpoint(
    ckpt: dict, device: Union[str, torch.device] = "cpu"
) -> Tuple[SAPModel, FamilyConfig, ModelWidths, dict]:
    """Rebuild a model from a checkpoint dict, verifying that the stored
    widths match the widths implied by the actual weight shapes."""
    family = FamilyConfig.from_dict(ckpt["family"])
    widths = ModelWidths.from_dict(ckpt["widths"])
    derived = widths_from_state_dict(ckpt["model_state"], family)
    if derived.to_dict() != widths.to_dict():
        raise ValueError(
            "checkpoint widths metadata does not match the weight shapes: "
            f"stored={widths.to_dict()} derived={derived.to_dict()}"
        )
    model = SAPModel(family, widths)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    return model, family, widths, ckpt.get("meta", {})


def load_model(path: Union[str, Path], device: Union[str, torch.device] = "cpu"):
    """Convenience: path -> (model, family, widths, meta)."""
    return model_from_checkpoint(load_checkpoint(path), device=device)
