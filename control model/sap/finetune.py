"""Supervised fine-tuning of any family checkpoint for sequence classification.

Why this exists
---------------
A pretrained LM can only be compared by perplexity, and perplexity alone is a
weak instrument: it is sensitive to tokenizer and corpus, and it says nothing
about whether the representations are *usable*. Downstream classification
accuracy is the standard second instrument, and it is the one a committee will
ask for. This module fine-tunes the merged model and the conventional baseline
under identical conditions so their numbers are directly comparable.

Architecture
------------
The backbone is an unmodified `SAPModel`; this module never touches the merge
math. A fresh linear head maps the final hidden state to class logits:

    input_ids -> embed -> L blocks -> final_norm -> [pool] -> dropout -> Linear(d_model, C)

Pooling is the hidden state at the LAST REAL TOKEN of each sequence, which is
the standard choice for a causal LM (a decoder-only model only "sees" the whole
sequence at its final position).

Padding, and why no attention mask is needed
--------------------------------------------
Sequences are RIGHT-padded. Attention is causal, so position t attends only to
positions <= t; the pad tokens sit strictly to the right of the last real token
and therefore cannot influence the pooled position. Right padding + last-real-
token pooling is exactly correct under a causal mask, which is why this module
adds no padding mask (adding one would change nothing).

Fair-comparison contract
------------------------
Fine-tuning is noisy at this scale. Everything that could differ between the
merged model and the baseline is pinned: same dataset, same tokenization cache,
same hyperparameters, same seed list, same budget. The ONLY thing that varies is
which checkpoint the backbone came from. Run >= 3 seeds and report mean +- std;
a single-seed difference of one or two points is not a result.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FamilyConfig, ModelWidths
from .model import SAPModel, atomic_torch_save, load_checkpoint, model_from_checkpoint

FINETUNE_FORMAT_VERSION = 1


# ===========================================================================
# Dataset registry
# ===========================================================================

@dataclass
class DatasetSpec:
    """Everything needed to turn a named dataset into (texts, labels) splits."""
    hf_name: str
    hf_config: Optional[str] = None
    text_key: str = "text"
    label_key: str = "label"
    train_split: str = "train"
    eval_split: str = "test"
    num_labels: int = 2
    label_names: Optional[List[str]] = None
    description: str = ""
    # when the dataset has no dedicated validation split, carve one from train
    carve_val_from_train: float = 0.0


DATASETS: Dict[str, DatasetSpec] = {
    "ag_news": DatasetSpec(
        hf_name="ag_news", text_key="text", label_key="label",
        train_split="train", eval_split="test", num_labels=4,
        label_names=["World", "Sports", "Business", "Sci/Tech"],
        carve_val_from_train=0.05,
        description="Topic classification, 120k train / 7.6k test, 4 balanced classes. "
                    "RECOMMENDED DEFAULT: large enough to be low-variance, 4-way so "
                    "chance is 25%, and topic knowledge is exactly what web/edu "
                    "pretraining should confer.",
    ),
    "sst2": DatasetSpec(
        hf_name="glue", hf_config="sst2", text_key="sentence", label_key="label",
        train_split="train", eval_split="validation", num_labels=2,
        label_names=["negative", "positive"],
        carve_val_from_train=0.0,
        description="Binary sentiment on short movie-review sentences, 67k train / "
                    "872 validation. Harder and noisier than AG News; the small "
                    "eval split makes single-seed differences unreliable.",
    ),
    "imdb": DatasetSpec(
        hf_name="imdb", text_key="text", label_key="label",
        train_split="train", eval_split="test", num_labels=2,
        label_names=["negative", "positive"],
        carve_val_from_train=0.05,
        description="Binary sentiment on full-length reviews, 25k train / 25k test. "
                    "Documents are long, so results depend on --max-length; good for "
                    "testing whether long-context representations survived merging.",
    ),
    "dbpedia_14": DatasetSpec(
        hf_name="dbpedia_14", text_key="content", label_key="label",
        train_split="train", eval_split="test", num_labels=14,
        carve_val_from_train=0.02,
        description="14-way ontology classification, 560k train. Very easy (strong "
                    "models saturate above 98%), so it compresses differences — use "
                    "as a sanity check, not as the headline benchmark.",
    ),
    "trec": DatasetSpec(
        hf_name="trec", text_key="text", label_key="coarse_label",
        train_split="train", eval_split="test", num_labels=6,
        carve_val_from_train=0.1,
        description="Question-type classification, 5.5k train / 500 test. Tiny and "
                    "high-variance; useful only as a low-resource stress test.",
    ),
}


def describe_datasets() -> str:
    lines = ["Built-in datasets (--dataset NAME):", ""]
    for name, s in DATASETS.items():
        lines.append(f"  {name}  ({s.num_labels} classes)")
        for chunk in _wrap(s.description, 72):
            lines.append(f"      {chunk}")
        lines.append("")
    lines.append("Any other HuggingFace dataset:")
    lines.append("  --dataset hf:NAME[:CONFIG] --num-labels K "
                 "--text-key text --label-key label")
    lines.append("Local files (.jsonl or .csv with a text column and a label column):")
    lines.append("  --dataset file:train.jsonl --eval-file test.jsonl --num-labels K")
    return "\n".join(lines)


def _wrap(text: str, width: int) -> List[str]:
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


def resolve_spec(
    dataset: str,
    num_labels: Optional[int] = None,
    text_key: Optional[str] = None,
    label_key: Optional[str] = None,
    train_split: Optional[str] = None,
    eval_split: Optional[str] = None,
) -> DatasetSpec:
    """Named dataset, `hf:NAME[:CONFIG]`, or `file:PATH` -> DatasetSpec."""
    if dataset in DATASETS:
        spec = dataclasses.replace(DATASETS[dataset])
    elif dataset.startswith("hf:"):
        parts = dataset.split(":")
        spec = DatasetSpec(
            hf_name=parts[1],
            hf_config=parts[2] if len(parts) > 2 and parts[2] else None,
            num_labels=num_labels or 2,
        )
    elif dataset.startswith("file:"):
        spec = DatasetSpec(hf_name=dataset, num_labels=num_labels or 2)
    else:
        raise ValueError(
            f"unknown dataset {dataset!r}.\n\n{describe_datasets()}"
        )
    if num_labels is not None:
        spec.num_labels = num_labels
    if text_key:
        spec.text_key = text_key
    if label_key:
        spec.label_key = label_key
    if train_split:
        spec.train_split = train_split
    if eval_split:
        spec.eval_split = eval_split
    return spec


# ===========================================================================
# Loading raw (text, label) pairs
# ===========================================================================

def _load_local(path: str, text_key: str, label_key: str) -> Tuple[List[str], List[int]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    texts, labels = [], []
    if p.suffix == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                texts.append(str(row[text_key]))
                labels.append(int(row[label_key]))
    elif p.suffix == ".csv":
        import csv
        with open(p, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                texts.append(str(row[text_key]))
                labels.append(int(row[label_key]))
    else:
        raise ValueError(f"unsupported local file type: {p.suffix} (use .jsonl or .csv)")
    return texts, labels


def load_splits(
    spec: DatasetSpec,
    train_file: Optional[str] = None,
    eval_file: Optional[str] = None,
    max_train: Optional[int] = None,
    max_eval: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Tuple[List[str], List[int]]]:
    """Return {'train': (texts, labels), 'eval': (texts, labels)}."""
    if spec.hf_name.startswith("file:") or train_file:
        tf = train_file or spec.hf_name[len("file:"):]
        if not eval_file:
            raise ValueError("local datasets need --eval-file as well as the train file")
        tr = _load_local(tf, spec.text_key, spec.label_key)
        ev = _load_local(eval_file, spec.text_key, spec.label_key)
    else:
        from datasets import load_dataset  # lazy: only needed for hf datasets
        ds_tr = load_dataset(spec.hf_name, spec.hf_config, split=spec.train_split)
        ds_ev = load_dataset(spec.hf_name, spec.hf_config, split=spec.eval_split)
        tr = ([str(t) for t in ds_tr[spec.text_key]],
              [int(l) for l in ds_tr[spec.label_key]])
        ev = ([str(t) for t in ds_ev[spec.text_key]],
              [int(l) for l in ds_ev[spec.label_key]])

    # deterministic subsampling (used for quick smoke runs)
    def cut(pair, n):
        if n is None or n >= len(pair[0]):
            return pair
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(pair[0]))[:n]
        return ([pair[0][i] for i in idx], [pair[1][i] for i in idx])

    return {"train": cut(tr, max_train), "eval": cut(ev, max_eval)}


# ===========================================================================
# Tokenization (must match the family tokenizer exactly)
# ===========================================================================

class FamilyTokenizer:
    """Thin wrapper so the fine-tuning data uses the SAME token ids the model
    was pretrained on. Defaults to tiktoken, matching scripts/prepare_data.py's
    recommended backend."""

    def __init__(self, name: str = "gpt2", backend: str = "tiktoken"):
        self.name, self.backend = name, backend
        if backend == "tiktoken":
            import tiktoken
            self.enc = tiktoken.get_encoding(name)
            self.vocab_size = self.enc.n_vocab
            self.pad_id = self.enc.eot_token
        elif backend == "hf":
            import os
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from transformers import AutoTokenizer
            self.enc = AutoTokenizer.from_pretrained(name)
            self.enc.model_max_length = int(1e12)
            self.vocab_size = len(self.enc)
            self.pad_id = self.enc.eos_token_id
        else:
            raise ValueError(f"unknown tokenizer backend {backend!r}")

    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        if self.backend == "tiktoken":
            return self.enc.encode_ordinary_batch(list(texts))
        return self.enc(list(texts), add_special_tokens=False)["input_ids"]


def encode_split(
    tokenizer: FamilyTokenizer,
    texts: Sequence[str],
    labels: Sequence[int],
    max_length: int,
    batch: int = 512,
) -> Dict[str, np.ndarray]:
    """Encode, truncate to the LAST `max_length` tokens, right-pad.

    Truncation keeps the TAIL of each document, not the head: the pooled
    representation is taken at the final position, so the tokens nearest the
    pooling point are the informative ones.
    """
    n = len(texts)
    ids = np.full((n, max_length), tokenizer.pad_id, dtype=np.int32)
    lens = np.zeros(n, dtype=np.int32)
    for lo in range(0, n, batch):
        enc = tokenizer.encode_batch(texts[lo: lo + batch])
        for j, seq in enumerate(enc):
            if not seq:
                seq = [tokenizer.pad_id]
            seq = seq[-max_length:]
            i = lo + j
            ids[i, : len(seq)] = seq
            lens[i] = len(seq)
    return {"input_ids": ids, "lengths": lens,
            "labels": np.asarray(labels, dtype=np.int64)}


def dataset_tag(dataset: str) -> str:
    """Filesystem-safe short label for a dataset spec.

    `file:` specs carry a full path (drive letters, separators) and `hf:` specs
    carry slashes and colons; both would otherwise create nested or invalid
    paths when used in a filename. A 6-hex-digit suffix keeps two specs that
    sanitize to the same string distinct.
    """
    if dataset.startswith("file:"):
        base = Path(dataset[len("file:"):]).stem
    elif dataset.startswith("hf:"):
        base = dataset[len("hf:"):]
    else:
        base = dataset
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-") or "dataset"
    digest = hashlib.md5(dataset.encode("utf-8")).hexdigest()[:6]
    return f"{safe[:40]}_{digest}"


def cache_key(dataset: str, tokenizer: str, backend: str, max_length: int,
              max_train: Optional[int], max_eval: Optional[int]) -> str:
    return (f"{dataset_tag(dataset)}__{tokenizer}_{backend}__len{max_length}"
            f"__tr{max_train}__ev{max_eval}")


def build_or_load_cache(
    cache_dir: Union[str, Path],
    key: str,
    build,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Tokenizing 120k documents takes minutes; do it once per configuration.

    Sharing one cache across the merged model and the baseline also guarantees
    they are trained and evaluated on byte-identical inputs."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.npz"
    if path.exists():
        z = np.load(path)
        out = {}
        for split in ("train", "eval"):
            out[split] = {k: z[f"{split}_{k}"] for k in ("input_ids", "lengths", "labels")}
        print(f"[data] loaded tokenized cache {path}")
        return out
    data = build()
    flat = {}
    for split, d in data.items():
        for k, v in d.items():
            flat[f"{split}_{k}"] = v
    np.savez(path, **flat)
    print(f"[data] wrote tokenized cache {path}")
    return data


# ===========================================================================
# The classification model
# ===========================================================================

class SAPClassifier(nn.Module):
    """Family backbone + linear classification head over the pooled final state."""

    def __init__(self, backbone: SAPModel, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(backbone.family.d_model, num_labels, bias=True)
        nn.init.normal_(self.score.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.score.bias)

    @property
    def family(self) -> FamilyConfig:
        return self.backbone.family

    def hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the backbone stack and return the post-final-norm states.

        Deliberately re-implements SAPModel.forward's body instead of calling it,
        because we need the hidden states rather than the vocabulary logits —
        and materializing a (B, T, 50257) logit tensor here would waste GBs.
        """
        bb = self.backbone
        B, T = input_ids.shape
        if T > bb.family.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {bb.family.max_seq_len}")
        x = bb.embed(input_ids)
        cos = bb.rope_cos[:T].to(x.dtype)
        sin = bb.rope_sin[:T].to(x.dtype)
        for blk in bb.blocks:
            x = blk(x, cos, sin)
        return bb.final_norm(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.hidden_states(input_ids)                       # (B, T, d_model)
        # pool at the last REAL token; right padding + causal attention means
        # the pad positions cannot have influenced this position
        idx = (lengths - 1).clamp(min=0).long()
        pooled = h[torch.arange(h.size(0), device=h.device), idx]   # (B, d_model)
        logits = self.score(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.float(), labels)
        return logits, loss

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def trainable_parameter_count(self) -> Tuple[int, int]:
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.parameters())
        return tr, tot


# ===========================================================================
# Metrics
# ===========================================================================

def classification_metrics(preds: np.ndarray, labels: np.ndarray, num_labels: int) -> dict:
    """Accuracy plus macro-F1, computed without a sklearn dependency."""
    acc = float((preds == labels).mean())
    f1s, per_class = [], {}
    for c in range(num_labels):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        per_class[str(c)] = {"precision": prec, "recall": rec, "f1": f1,
                             "support": int((labels == c).sum())}
    return {"accuracy": acc, "macro_f1": float(np.mean(f1s)), "per_class": per_class}


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class FinetuneConfig:
    # identity / io
    name: str
    out_dir: str
    init_from: str                       # backbone checkpoint (merged / baseline / shard)

    # data
    dataset: str = "ag_news"
    num_labels: Optional[int] = None
    text_key: Optional[str] = None
    label_key: Optional[str] = None
    train_split: Optional[str] = None
    eval_split: Optional[str] = None
    train_file: Optional[str] = None
    eval_file: Optional[str] = None
    max_train: Optional[int] = None
    max_eval: Optional[int] = None
    max_length: int = 256
    cache_dir: str = "data/finetune_cache"
    tokenizer: str = "gpt2"
    tokenizer_backend: str = "tiktoken"

    # optimization
    seed: int = 0
    batch_size: int = 16
    grad_accum: int = 1
    lr: float = 2e-5                     # backbone LR (pretrained weights: small)
    head_lr: float = 1e-3                # fresh head LR (random init: large)
    min_lr_frac: float = 0.1
    warmup_frac: float = 0.06
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    dropout: float = 0.1
    freeze_backbone: bool = False        # True == linear probe

    # budget
    max_epochs: Optional[int] = 3
    max_hours: Optional[float] = None
    max_steps: Optional[int] = None

    # checkpointing
    checkpoint_every_min: Optional[float] = 20.0
    fresh: bool = False
    save_best: bool = True

    # logging / eval
    log_interval: int = 50
    eval_per_epoch: int = 1              # evaluations per epoch
    eval_batch_size: int = 32

    # hardware
    device: str = "auto"
    dtype: str = "auto"

    def resolved_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


RESUME_CRITICAL_FIELDS = (
    "init_from", "dataset", "max_length", "batch_size", "grad_accum",
    "seed", "freeze_backbone",
)


# ===========================================================================
# Evaluation
# ===========================================================================

@torch.no_grad()
def evaluate_classifier(
    model: SAPClassifier,
    data: Dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
) -> dict:
    model.eval()
    n = len(data["labels"])
    all_preds = np.zeros(n, dtype=np.int64)
    total_loss, total_n = 0.0, 0
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        x = torch.from_numpy(data["input_ids"][lo:hi].astype(np.int64)).to(device)
        L = torch.from_numpy(data["lengths"][lo:hi].astype(np.int64)).to(device)
        y = torch.from_numpy(data["labels"][lo:hi]).to(device)
        ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
               if amp_dtype is not None and device.type == "cuda"
               else _null())
        with ctx:
            logits, loss = model(x, L, labels=y)
        all_preds[lo:hi] = logits.float().argmax(-1).cpu().numpy()
        total_loss += float(loss) * (hi - lo)
        total_n += hi - lo
    model.train()
    m = classification_metrics(all_preds, data["labels"], model.num_labels)
    m["loss"] = total_loss / max(1, total_n)
    m["n"] = int(n)
    return m


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ===========================================================================
# Checkpoint I/O
# ===========================================================================

def save_finetune_checkpoint(
    path: Union[str, Path],
    model: SAPClassifier,
    meta: dict,
    optimizer_state: Optional[dict] = None,
    train_state: Optional[dict] = None,
    train_args: Optional[dict] = None,
) -> None:
    payload = {
        "format_version": FINETUNE_FORMAT_VERSION,
        "kind": "sap_finetune_classifier",
        "family": model.backbone.family.to_dict(),
        "widths": model.backbone.widths.to_dict(),
        "num_labels": model.num_labels,
        "backbone_state": model.backbone.state_dict(),
        "head_state": model.score.state_dict(),
        "meta": meta,
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    if train_state is not None:
        payload["train_state"] = train_state
    if train_args is not None:
        payload["train_args"] = train_args
    atomic_torch_save(payload, path)


def load_finetuned(path: Union[str, Path], device: Union[str, torch.device] = "cpu"
                   ) -> Tuple[SAPClassifier, dict]:
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    if ck.get("kind") != "sap_finetune_classifier":
        raise ValueError(f"{path} is not a SAP fine-tuned classifier checkpoint")
    family = FamilyConfig.from_dict(ck["family"])
    widths = ModelWidths.from_dict(ck["widths"])
    backbone = SAPModel(family, widths)
    backbone.load_state_dict(ck["backbone_state"])
    model = SAPClassifier(backbone, ck["num_labels"])
    model.score.load_state_dict(ck["head_state"])
    model.to(device)
    return model, ck.get("meta", {})


# ===========================================================================
# The fine-tuning loop
# ===========================================================================

def run_finetune(cfg: FinetuneConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.resolved_device()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # -- data ---------------------------------------------------------------
    spec = resolve_spec(cfg.dataset, cfg.num_labels, cfg.text_key, cfg.label_key,
                        cfg.train_split, cfg.eval_split)
    tokenizer = FamilyTokenizer(cfg.tokenizer, cfg.tokenizer_backend)
    key = cache_key(cfg.dataset, cfg.tokenizer, cfg.tokenizer_backend,
                    cfg.max_length, cfg.max_train, cfg.max_eval)

    def build():
        print(f"[data] loading {spec.hf_name}"
              f"{':' + spec.hf_config if spec.hf_config else ''} ...")
        splits = load_splits(spec, cfg.train_file, cfg.eval_file,
                             cfg.max_train, cfg.max_eval, seed=cfg.seed)
        out = {}
        for name, (texts, labels) in splits.items():
            print(f"[data] tokenizing {name}: {len(texts):,} examples "
                  f"(max_length={cfg.max_length})")
            out[name] = encode_split(tokenizer, texts, labels, cfg.max_length)
        return out

    data = build_or_load_cache(cfg.cache_dir, key, build)
    train_d, eval_d = data["train"], data["eval"]
    num_labels = spec.num_labels

    observed = int(train_d["labels"].max()) + 1
    if observed > num_labels:
        raise ValueError(
            f"dataset has at least {observed} distinct labels but num_labels="
            f"{num_labels}; pass --num-labels {observed}"
        )

    # -- model --------------------------------------------------------------
    latest_path = out_dir / "latest.pt"
    resuming = latest_path.exists() and not cfg.fresh

    if resuming:
        ck = torch.load(str(latest_path), map_location="cpu", weights_only=False)
        saved = ck.get("train_args", {})
        for f in RESUME_CRITICAL_FIELDS:
            now, then = getattr(cfg, f), saved.get(f)
            if then is not None and str(then) != str(now):
                raise ValueError(
                    f"resume mismatch on '{f}': checkpoint has {then!r}, command has "
                    f"{now!r}. Use --fresh to restart deliberately."
                )
        family = FamilyConfig.from_dict(ck["family"])
        widths = ModelWidths.from_dict(ck["widths"])
        backbone = SAPModel(family, widths)
        backbone.load_state_dict(ck["backbone_state"])
        model = SAPClassifier(backbone, ck["num_labels"], dropout=cfg.dropout)
        model.score.load_state_dict(ck["head_state"])
        backbone_source = ck.get("meta", {}).get("backbone_source", cfg.init_from)
    else:
        bb_ckpt = load_checkpoint(cfg.init_from)
        backbone, family, widths, bb_meta = model_from_checkpoint(bb_ckpt, device="cpu")
        model = SAPClassifier(backbone, num_labels, dropout=cfg.dropout)
        backbone_source = cfg.init_from
        print(f"[init] backbone from {cfg.init_from} "
              f"(role={bb_meta.get('role', '?')}, name={bb_meta.get('name', '?')}, "
              f"pretrain_tokens={bb_meta.get('tokens_seen', '?')})")
        ck = None

    model.set_backbone_trainable(not cfg.freeze_backbone)
    model.to(device)
    n_train_p, n_total_p = model.trainable_parameter_count()
    mode = "LINEAR PROBE (backbone frozen)" if cfg.freeze_backbone else "FULL FINE-TUNE"
    print(f"[{cfg.name}] {mode}: {n_train_p / 1e6:.2f}M trainable / "
          f"{n_total_p / 1e6:.1f}M total on {device}")

    # -- optimizer: two groups, because a fresh head needs a much larger LR --
    head_params = list(model.score.parameters())
    bb_decay, bb_nodecay = [], []
    for p in model.backbone.parameters():
        if p.requires_grad:
            (bb_decay if p.dim() >= 2 else bb_nodecay).append(p)
    groups = [{"params": head_params, "lr": cfg.head_lr, "weight_decay": 0.0,
               "tag": "head"}]
    if bb_decay:
        groups.append({"params": bb_decay, "lr": cfg.lr,
                       "weight_decay": cfg.weight_decay, "tag": "backbone_decay"})
    if bb_nodecay:
        groups.append({"params": bb_nodecay, "lr": cfg.lr, "weight_decay": 0.0,
                       "tag": "backbone_nodecay"})
    optimizer = torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.999))
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    # -- schedule -----------------------------------------------------------
    n = len(train_d["labels"])
    batches_per_epoch = max(1, n // cfg.batch_size)
    steps_per_epoch = max(1, batches_per_epoch // cfg.grad_accum)
    total_steps = cfg.max_steps
    if total_steps is None and cfg.max_epochs is not None:
        total_steps = steps_per_epoch * cfg.max_epochs
    warmup_steps = int(cfg.warmup_frac * total_steps) if total_steps else 0

    def lr_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        if not total_steps:
            return 1.0
        frac = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        frac = min(1.0, max(0.0, frac))
        return cfg.min_lr_frac + 0.5 * (1 - cfg.min_lr_frac) * (1 + math.cos(math.pi * frac))

    # -- training state -----------------------------------------------------
    step, epoch, cursor, elapsed_prev = 0, 0, 0, 0.0
    best = {"accuracy": -1.0}
    history: List[dict] = []
    if resuming:
        ts = ck["train_state"]
        optimizer.load_state_dict(ck["optimizer_state"])
        step, epoch, cursor = ts["step"], ts["epoch"], ts["cursor"]
        elapsed_prev = ts["elapsed_seconds"]
        best = ts.get("best", best)
        history = ts.get("history", [])
        print(f"[{cfg.name}] resumed: step {step}, epoch {epoch}, "
              f"best acc {best.get('accuracy', -1):.4f}, "
              f"{elapsed_prev / 60:.1f} min elapsed")

    use_bf16 = (cfg.dtype == "bf16"
                or (cfg.dtype == "auto" and device.type == "cuda"
                    and torch.cuda.is_bf16_supported()))
    amp_dtype = torch.bfloat16 if use_bf16 else None

    session_start = time.monotonic()

    def elapsed_total() -> float:
        return elapsed_prev + (time.monotonic() - session_start)

    def epoch_order(e: int) -> np.ndarray:
        return np.random.RandomState(cfg.seed + 1000 * e).permutation(n)

    def build_meta() -> dict:
        return {
            "role": "finetuned",
            "name": cfg.name,
            "backbone_source": backbone_source,
            "dataset": cfg.dataset,
            "num_labels": num_labels,
            "mode": "linear_probe" if cfg.freeze_backbone else "full_finetune",
            "seed": cfg.seed,
            "step": step,
            "epoch": epoch,
            "best": best,
            "elapsed_seconds": elapsed_total(),
        }

    args_dict = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}

    def save(path: Path, with_state: bool = True) -> None:
        ts = None
        opt = None
        if with_state:
            ts = {"step": step, "epoch": epoch, "cursor": cursor,
                  "elapsed_seconds": elapsed_total(), "best": best,
                  "history": history}
            opt = optimizer.state_dict()
        save_finetune_checkpoint(path, model, build_meta(), optimizer_state=opt,
                                 train_state=ts, train_args=args_dict)

    log_path = out_dir / "finetune_log.jsonl"

    def log(rec: dict) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    eval_every = max(1, steps_per_epoch // max(1, cfg.eval_per_epoch))
    print(f"[{cfg.name}] {n:,} train / {len(eval_d['labels']):,} eval examples | "
          f"{steps_per_epoch:,} steps/epoch | total {total_steps} steps | "
          f"eval every {eval_every} steps")

    last_ckpt = time.monotonic()
    ckpt_interval = (cfg.checkpoint_every_min or 0) * 60.0
    stop_reason = None
    order = epoch_order(epoch)
    running, running_n = 0.0, 0

    model.train()
    while True:
        if cfg.max_steps is not None and step >= cfg.max_steps:
            stop_reason = "max_steps"
            break
        if cfg.max_epochs is not None and epoch >= cfg.max_epochs:
            stop_reason = "max_epochs"
            break
        if cfg.max_hours is not None and elapsed_total() >= cfg.max_hours * 3600:
            stop_reason = "max_hours"
            break

        scale = lr_scale(step)
        for g, base in zip(optimizer.param_groups, base_lrs):
            g["lr"] = base * scale

        step_loss = 0.0
        for _ in range(cfg.grad_accum):
            if cursor >= batches_per_epoch:
                epoch += 1
                cursor = 0
                order = epoch_order(epoch)
                if cfg.max_epochs is not None and epoch >= cfg.max_epochs:
                    break
            sel = order[cursor * cfg.batch_size: (cursor + 1) * cfg.batch_size]
            cursor += 1
            x = torch.from_numpy(train_d["input_ids"][sel].astype(np.int64)).to(device)
            L = torch.from_numpy(train_d["lengths"][sel].astype(np.int64)).to(device)
            y = torch.from_numpy(train_d["labels"][sel]).to(device)
            ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                   if amp_dtype is not None and device.type == "cuda" else _null())
            with ctx:
                _, loss = model(x, L, labels=y)
            (loss / cfg.grad_accum).backward()
            step_loss += float(loss.detach()) / cfg.grad_accum

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        running += step_loss
        running_n += 1

        if step % cfg.log_interval == 0:
            avg = running / max(1, running_n)
            print(f"[{cfg.name}] step {step:>6} | loss {avg:.4f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.2e} | epoch {epoch} | "
                  f"{elapsed_total() / 60:.1f} min")
            log({"step": step, "train_loss": avg, "epoch": epoch,
                 "elapsed_min": elapsed_total() / 60})
            running, running_n = 0.0, 0

        if step % eval_every == 0:
            m = evaluate_classifier(model, eval_d, cfg.eval_batch_size, device, amp_dtype)
            rec = {"step": step, "epoch": epoch, "eval_loss": m["loss"],
                   "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
            history.append(rec)
            log(rec)
            print(f"[{cfg.name}] step {step:>6} | EVAL acc {m['accuracy']:.4f} | "
                  f"macro-F1 {m['macro_f1']:.4f} | loss {m['loss']:.4f}")
            if m["accuracy"] > best.get("accuracy", -1):
                best = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                        "loss": m["loss"], "step": step, "epoch": epoch}
                if cfg.save_best:
                    save(out_dir / "best.pt", with_state=False)
                    print(f"[{cfg.name}]   new best -> {out_dir / 'best.pt'}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if ckpt_interval and time.monotonic() - last_ckpt >= ckpt_interval:
            save(latest_path)
            last_ckpt = time.monotonic()
            print(f"[{cfg.name}] checkpoint saved at step {step}")

    # -- final ---------------------------------------------------------------
    save(latest_path)
    final_metrics = evaluate_classifier(model, eval_d, cfg.eval_batch_size,
                                        device, amp_dtype)
    save(out_dir / "final.pt", with_state=False)

    summary = {
        "name": cfg.name,
        "backbone_source": backbone_source,
        "dataset": cfg.dataset,
        "mode": "linear_probe" if cfg.freeze_backbone else "full_finetune",
        "seed": cfg.seed,
        "num_labels": num_labels,
        "stop_reason": stop_reason,
        "steps": step,
        "epochs": epoch,
        "elapsed_hours": elapsed_total() / 3600,
        "final": {k: final_metrics[k] for k in ("accuracy", "macro_f1", "loss", "n")},
        "best": best,
        "history": history,
        "trainable_params": n_train_p,
        "total_params": n_total_p,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log({"event": "finished", **{k: v for k, v in summary.items() if k != "history"}})

    print(f"\n[{cfg.name}] done ({stop_reason}) in {summary['elapsed_hours']:.2f}h")
    print(f"  final : acc {final_metrics['accuracy']:.4f}  "
          f"macro-F1 {final_metrics['macro_f1']:.4f}")
    print(f"  best  : acc {best['accuracy']:.4f}  macro-F1 {best['macro_f1']:.4f} "
          f"(step {best['step']})")
    print(f"  summary -> {out_dir / 'summary.json'}")
    return summary
