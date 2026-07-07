"""Family skeleton and per-model width configuration.

The *family* is the contract every model must satisfy so that merging is
well-defined: number of layers, residual width, head dimension, tokenizer
vocabulary, RoPE base, and norm epsilon are fixed for the whole project.
The *widths* (heads per layer, FFN neurons per layer) are free per model
and are what merging adds up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Union


# ---------------------------------------------------------------------------
# Family skeleton (fixed across every model that will ever be merged)
# ---------------------------------------------------------------------------

@dataclass
class FamilyConfig:
    n_layers: int          # L — transformer blocks; block i of A merges with block i of B
    d_model: int           # residual stream width — the shared bus all units read/write
    d_head: int            # per-head dimension — uniform so heads stack cleanly
    vocab_size: int        # must equal the tokenizer's vocab; embeddings are averaged
    max_seq_len: int       # context length; RoPE cache is precomputed to this length
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.d_model % self.d_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by d_head ({self.d_head}) "
                "so that a balanced seed configuration exists."
            )
        for name in ("n_layers", "d_model", "d_head", "vocab_size", "max_seq_len"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

    # -- (de)serialization ---------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FamilyConfig":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "FamilyConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save_json(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def __eq__(self, other) -> bool:
        if not isinstance(other, FamilyConfig):
            return NotImplemented
        return self.to_dict() == other.to_dict()


# ---------------------------------------------------------------------------
# Per-model widths (free to differ between family members)
# ---------------------------------------------------------------------------

@dataclass
class ModelWidths:
    """Per-layer head counts and FFN widths.

    Stored as lists of length n_layers so that merged models (whose widths
    are per-layer sums) and hand-crafted heterogeneous models are all
    representable in one format.
    """
    n_heads: List[int] = field(default_factory=list)
    d_ff: List[int] = field(default_factory=list)

    def __post_init__(self):
        if len(self.n_heads) != len(self.d_ff):
            raise ValueError("n_heads and d_ff lists must have the same length")
        if any(h <= 0 for h in self.n_heads) or any(f <= 0 for f in self.d_ff):
            raise ValueError("all widths must be positive")

    @property
    def n_layers(self) -> int:
        return len(self.n_heads)

    @classmethod
    def uniform(cls, n_layers: int, n_heads: int, d_ff: int) -> "ModelWidths":
        return cls(n_heads=[n_heads] * n_layers, d_ff=[d_ff] * n_layers)

    @classmethod
    def from_spec(cls, n_layers: int, heads_spec: str, dff_spec: str) -> "ModelWidths":
        """Parse CLI specs: either a single int ('4') applied to every layer,
        or a comma-separated per-layer list ('4,4,6,6,...')."""
        def parse(spec: str) -> List[int]:
            parts = [int(p) for p in str(spec).split(",")]
            if len(parts) == 1:
                return parts * n_layers
            if len(parts) != n_layers:
                raise ValueError(
                    f"width spec '{spec}' has {len(parts)} entries but the family "
                    f"has {n_layers} layers"
                )
            return parts
        return cls(n_heads=parse(heads_spec), d_ff=parse(dff_spec))

    def to_dict(self) -> dict:
        return {"n_heads": list(self.n_heads), "d_ff": list(self.d_ff)}

    @classmethod
    def from_dict(cls, d: dict) -> "ModelWidths":
        return cls(n_heads=list(d["n_heads"]), d_ff=list(d["d_ff"]))


def merged_widths(widths_list: List[ModelWidths]) -> ModelWidths:
    """Widths of an N-way merged model: per-layer sums of parent widths."""
    L = widths_list[0].n_layers
    for w in widths_list:
        if w.n_layers != L:
            raise ValueError("all parents must have the same number of layers")
    return ModelWidths(
        n_heads=[sum(w.n_heads[i] for w in widths_list) for i in range(L)],
        d_ff=[sum(w.d_ff[i] for w in widths_list) for i in range(L)],
    )
