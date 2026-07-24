"""Dataset preparation and deterministic batch sampling.

Design contract
---------------
* Raw text (any source) is tokenized ONCE with the family tokenizer and
  written as flat binary token streams (.bin files of uint16/uint32).
  Documents are separated by the tokenizer's EOS token.
* Partitioning D into D_1..D_N happens at the DOCUMENT level, at prepare
  time, deterministically (seeded hash of the document index), so the same
  command reproduces the same partitions bit-for-bit on any machine.
* A held-out validation split (val.bin) is carved out FIRST, before
  partitioning, so no model — seed, shard, or merged — ever trains on it.
* Training machines only need the .bin files + meta.json; the tokenizer is
  only needed for data prep and for text generation during eval.

Batch sampling is a deterministic permutation of non-overlapping
(seq_len+1)-token chunks, re-seeded per epoch, with an (epoch, cursor)
state that is saved into training checkpoints — this is what makes
crash-resume exact.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch

META_FILENAME = "meta.json"


# ---------------------------------------------------------------------------
# Binary token files
# ---------------------------------------------------------------------------

def dtype_for_vocab(vocab_size: int) -> str:
    return "uint16" if vocab_size <= 65536 else "uint32"


def find_meta(bin_path: Union[str, Path]) -> Optional[dict]:
    meta_path = Path(bin_path).resolve().parent / META_FILENAME
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_tokens(bin_path: Union[str, Path], dtype: Optional[str] = None) -> np.memmap:
    """Memory-map a token .bin. dtype comes from the sibling meta.json unless
    overridden explicitly."""
    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(bin_path)
    if dtype is None:
        meta = find_meta(bin_path)
        if meta is None:
            raise ValueError(
                f"no {META_FILENAME} found next to {bin_path}; pass dtype explicitly "
                "(uint16 or uint32)"
            )
        dtype = meta["dtype"]
    return np.memmap(str(bin_path), dtype=np.dtype(dtype), mode="r")


class BinWriter:
    """Buffered append-only writer for a token .bin file."""

    def __init__(self, path: Union[str, Path], dtype: str, flush_tokens: int = 1_000_000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dtype = np.dtype(dtype)
        self.flush_tokens = flush_tokens
        self._buf: List[np.ndarray] = []
        self._buffered = 0
        self.token_count = 0
        self.doc_count = 0
        self._fh = open(self.path, "wb")

    def add(self, token_ids: List[int]) -> None:
        arr = np.asarray(token_ids, dtype=self.dtype)
        self._buf.append(arr)
        self._buffered += len(arr)
        self.token_count += len(arr)
        self.doc_count += 1
        if self._buffered >= self.flush_tokens:
            self._flush()

    def _flush(self) -> None:
        if self._buf:
            np.concatenate(self._buf).tofile(self._fh)
            self._buf = []
            self._buffered = 0

    def close(self) -> None:
        self._flush()
        self._fh.close()


# ---------------------------------------------------------------------------
# Document iteration (txt / jsonl / HuggingFace datasets)
# ---------------------------------------------------------------------------

def iter_documents(
    inputs: List[str],
    text_key: str = "text",
    txt_mode: str = "line",
) -> Iterator[Tuple[int, int, str]]:
    """Yield (doc_index, input_index, text) over all inputs, in order.

    Supported inputs:
      * path/to/file.txt    — txt_mode 'line': one document per non-empty line;
                              txt_mode 'file': the whole file is one document
      * path/to/file.jsonl  — one JSON object per line; text under `text_key`
      * hf:name[:config][:split] — a HuggingFace dataset, streamed
                              (e.g. hf:HuggingFaceFW/fineweb-edu:sample-10BT:train)
    """
    doc_index = 0
    for input_index, spec in enumerate(inputs):
        if spec.startswith("hf:"):
            parts = spec.split(":")
            name = parts[1]
            config = parts[2] if len(parts) > 2 and parts[2] else None
            split = parts[3] if len(parts) > 3 else "train"
            from datasets import load_dataset  # lazy: only needed for hf: inputs
            ds = load_dataset(name, config, split=split, streaming=True)
            for row in ds:
                text = row.get(text_key)
                if text:
                    yield doc_index, input_index, text
                    doc_index += 1
            continue

        path = Path(spec)
        if not path.exists():
            raise FileNotFoundError(spec)
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    text = json.loads(line).get(text_key)
                    if text:
                        yield doc_index, input_index, text
                        doc_index += 1
        elif path.suffix == ".txt":
            if txt_mode == "file":
                yield doc_index, input_index, path.read_text(encoding="utf-8")
                doc_index += 1
            else:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield doc_index, input_index, line
                            doc_index += 1
        else:
            raise ValueError(f"unsupported input type: {spec} (use .txt, .jsonl, or hf:)")


# ---------------------------------------------------------------------------
# Deterministic document routing
# ---------------------------------------------------------------------------

def _hash_unit(tag: str, seed: int, i: int) -> float:
    """Deterministic uniform-[0,1) value for document i under a given tag.

    crc32 is stable across platforms, Python versions and PYTHONHASHSEED,
    which is what guarantees that re-running prepare_data reproduces the
    exact same partitions anywhere."""
    return zlib.crc32(f"{tag}:{seed}:{i}".encode()) / 2**32


def route_document(
    doc_index: int,
    input_index: int,
    num_partitions: int,
    seed: int,
    val_fraction: float,
    mode: str,
    block_docs: int,
) -> Tuple[str, int]:
    """Decide where a document goes: ('val', -1) or ('part', k)."""
    if val_fraction > 0 and _hash_unit("val", seed, doc_index) < val_fraction:
        return "val", -1
    if mode == "random":
        k = zlib.crc32(f"part:{seed}:{doc_index}".encode()) % num_partitions
    elif mode == "blocks":
        k = (doc_index // block_docs) % num_partitions
    elif mode == "by-input":
        k = input_index % num_partitions
    else:
        raise ValueError(f"unknown partition mode: {mode}")
    return "part", k


# ---------------------------------------------------------------------------
# Dataset preparation (tokenize + partition + write bins)
# ---------------------------------------------------------------------------

def prepare_dataset(
    inputs: List[str],
    tokenizer_name: str,
    out_dir: Union[str, Path],
    num_partitions: int,
    tokenizer_backend: str = "hf",
    val_fraction: float = 0.005,
    seed_fraction: float = 0.0,
    partition_val_fraction: float = 0.0,
    mode: str = "random",
    block_docs: int = 10_000,
    seed: int = 1234,
    text_key: str = "text",
    txt_mode: str = "line",
    eos_id: Optional[int] = None,
    max_docs: Optional[int] = None,
    max_tokens: Optional[int] = None,
    max_doc_chars: Optional[int] = 2_000_000,
    encode_batch_docs: int = 512,
) -> dict:
    """Tokenize `inputs` and write:

        out_dir/val.bin              global held-out validation (never trained on)
        out_dir/seed.bin             mixed sample of ALL partitions (if seed_fraction > 0)
        out_dir/part_01.bin ...      the N training partitions D_1..D_N
        out_dir/part_01.val.bin ...  per-partition held-out (if partition_val_fraction > 0)
        out_dir/meta.json            tokenizer name, vocab, dtype, token counts

    The seed sample deliberately OVERLAPS the partitions: the seed model is
    supposed to see a mixed sample drawn from all of D before branching.
    """
    import os
    from tqdm import tqdm

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # `encode_batch(list[str]) -> list[list[int]]` is set up per backend below.
    if tokenizer_backend == "tiktoken":
        # tiktoken's GPT-2 BPE is byte-for-byte compatible with HF "gpt2"
        # (same 50257 vocab, same ids, eot=50256) but is far more stable — it
        # does not have the native segfaults the HF Rust tokenizer can hit on
        # long runs. This is what nanoGPT/llm.c use for FineWeb.
        import tiktoken
        enc = tiktoken.get_encoding(tokenizer_name)
        vocab_size = enc.n_vocab
        if eos_id is None:
            eos_id = enc.eot_token

        def encode_batch(docs):
            return enc.encode_ordinary_batch(docs)   # ignores special tokens

    elif tokenizer_backend == "hf":
        # Disable the fast-tokenizer's internal thread pool (a known source of
        # segfaults/hangs on long runs); single-threaded is slower but safer.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from transformers import AutoTokenizer  # lazy: only data prep needs it
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()         # silence the benign "N > 1024" warning
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.model_max_length = int(1e12)   # no implicit truncation / length warnings
        vocab_size = len(tokenizer)
        if eos_id is None:
            eos_id = tokenizer.eos_token_id

        def encode_batch(docs):
            return tokenizer(docs, add_special_tokens=False)["input_ids"]

    else:
        raise ValueError(
            f"unknown tokenizer_backend {tokenizer_backend!r} (use 'hf' or 'tiktoken')"
        )

    dtype = dtype_for_vocab(vocab_size)
    if eos_id is None:
        raise ValueError(
            "tokenizer has no EOS token; pass eos_id explicitly (documents must be "
            "separated by a delimiter token)"
        )

    writers: Dict[str, BinWriter] = {}

    def writer(name: str) -> BinWriter:
        if name not in writers:
            writers[name] = BinWriter(out_dir / name, dtype)
        return writers[name]

    doc_buf: List[str] = []
    route_buf: List[Tuple[str, int, int]] = []   # (kind, k, doc_index)

    def flush_batch() -> None:
        if not doc_buf:
            return
        encoded = encode_batch(doc_buf)
        for ids, (kind, k, di) in zip(encoded, route_buf):
            if not ids:
                continue
            ids = list(ids) + [eos_id]
            if kind == "val":
                writer("val.bin").add(ids)
                continue
            # training document: partition (or per-partition val), plus seed sample
            if partition_val_fraction > 0 and _hash_unit("pval", seed, di) < partition_val_fraction:
                writer(f"part_{k + 1:02d}.val.bin").add(ids)
            else:
                writer(f"part_{k + 1:02d}.bin").add(ids)
                if seed_fraction > 0 and _hash_unit("seed", seed, di) < seed_fraction:
                    writer("seed.bin").add(ids)
        doc_buf.clear()
        route_buf.clear()

    def total_tokens_written() -> int:
        return sum(w.token_count for w in writers.values())

    n_docs = 0
    stop = False
    for doc_index, input_index, text in tqdm(
        iter_documents(inputs, text_key=text_key, txt_mode=txt_mode),
        desc="tokenizing", unit=" docs",
    ):
        kind, k = route_document(
            doc_index, input_index, num_partitions, seed, val_fraction, mode, block_docs
        )
        # truncate pathologically long documents (rare web-scraped junk) before
        # tokenizing — a single multi-MB document can crash the native tokenizer
        if max_doc_chars is not None and len(text) > max_doc_chars:
            text = text[:max_doc_chars]
        doc_buf.append(text)
        route_buf.append((kind, k, doc_index))
        n_docs += 1
        if len(doc_buf) >= encode_batch_docs:
            flush_batch()
            # token cap is checked after a flush, so it stops within one batch
            # (~encode_batch_docs documents) of the requested budget
            if max_tokens is not None and total_tokens_written() >= max_tokens:
                stop = True
        if max_docs is not None and n_docs >= max_docs:
            stop = True
        if stop:
            break
    flush_batch()

    files = {}
    for name, w in sorted(writers.items()):
        w.close()
        files[name] = {"tokens": w.token_count, "documents": w.doc_count}

    meta = {
        "tokenizer": tokenizer_name,
        "vocab_size": vocab_size,
        "dtype": dtype,
        "eos_id": int(eos_id),
        "num_partitions": num_partitions,
        "partition_mode": mode,
        "val_fraction": val_fraction,
        "seed_fraction": seed_fraction,
        "partition_val_fraction": partition_val_fraction,
        "routing_seed": seed,
        "total_documents": n_docs,
        "total_tokens": total_tokens_written(),
        "files": files,
    }
    with open(out_dir / META_FILENAME, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


# ---------------------------------------------------------------------------
# Deterministic, resumable batch sampling
# ---------------------------------------------------------------------------

class ChunkSampler:
    """Epoch-based sampler over non-overlapping (seq_len+1)-token chunks.

    Each epoch is a fresh permutation of all chunks, seeded by
    (data_seed + epoch), so the full data order of any run is a pure
    function of (data file, seq_len, batch_size, data_seed) — no hidden
    state. Resuming from (epoch, cursor) reproduces the exact batches the
    interrupted run would have seen.
    """

    def __init__(
        self,
        n_tokens: int,
        seq_len: int,
        batch_size: int,
        data_seed: int,
        epoch: int = 0,
        cursor: int = 0,
    ):
        self.n_chunks = (n_tokens - 1) // seq_len
        if self.n_chunks < batch_size:
            raise ValueError(
                f"dataset too small: {self.n_chunks} chunks of {seq_len} tokens, "
                f"but batch_size is {batch_size}"
            )
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.data_seed = data_seed
        self.batches_per_epoch = self.n_chunks // batch_size   # drop last partial batch
        self.epoch = epoch
        self.cursor = cursor
        self._perm: Optional[np.ndarray] = None

    def _permutation(self) -> np.ndarray:
        if self._perm is None:
            rng = np.random.RandomState(self.data_seed + self.epoch)
            self._perm = rng.permutation(self.n_chunks)
        return self._perm

    def next_batch(self) -> np.ndarray:
        """Return the chunk indices of the next micro-batch, advancing state.
        Rolls into the next epoch eagerly, so `self.epoch` always equals the
        number of fully completed epochs."""
        perm = self._permutation()
        lo = self.cursor * self.batch_size
        idxs = perm[lo: lo + self.batch_size]
        self.cursor += 1
        if self.cursor >= self.batches_per_epoch:
            self.epoch += 1
            self.cursor = 0
            self._perm = None
        return idxs

    def state(self) -> dict:
        return {"epoch": self.epoch, "cursor": self.cursor}


def get_batch(
    tokens: np.memmap,
    chunk_idxs: np.ndarray,
    seq_len: int,
    device: Union[str, torch.device],
    pin: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialize (x, y) int64 tensors for the given chunk indices.
    y is x shifted one token right (next-token prediction).

    pin=False skips pinned-host-memory staging: marginally slower host->GPU
    copies, but avoids cudaHostRegister — useful on machines whose RAM/driver
    stack is suspect (pinned allocations are a classic native-crash site)."""
    xs = np.stack([tokens[i * seq_len: i * seq_len + seq_len] for i in chunk_idxs])
    ys = np.stack([tokens[i * seq_len + 1: i * seq_len + seq_len + 1] for i in chunk_idxs])
    x = torch.from_numpy(xs.astype(np.int64))
    y = torch.from_numpy(ys.astype(np.int64))
    if torch.device(device).type == "cuda":
        if pin:
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
