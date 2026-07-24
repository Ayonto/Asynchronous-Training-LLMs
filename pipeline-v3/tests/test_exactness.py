"""Exactness and reliability tests for the SAP pipeline.

These are the non-negotiable checks the thesis rests on:

  * merged sublayer == alpha-weighted average of parent sublayers (float64)
  * N-way merging, sequential == simultaneous, over-complete W_O
  * unscaled mode == exact function SUM
  * norm-gain absorption changes nothing
  * the exactness gate actually catches corrupted merges (negative test)
  * exact parameter accounting (the growth law, to the parameter)
  * function-preserving seed growth
  * checkpoint roundtrips and crash-resume produce identical training

All tests run on CPU in seconds:  python -m pytest tests -q
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

from sap.config import FamilyConfig, ModelWidths, merged_widths
from sap.data import ChunkSampler
from sap.merge import (
    MergeExactnessError,
    absorb_norm_gains,
    merge_models,
    resolve_alphas,
    verify_merge,
    weight_average,
)
from sap.model import (
    SAPModel,
    build_rope_cache,
    load_model,
    save_model_checkpoint,
    widths_from_state_dict,
)

FAM = FamilyConfig(n_layers=2, d_model=16, d_head=4, vocab_size=64, max_seq_len=32)
TOL64 = 1e-12   # float64 exactness threshold


def make_model(n_heads, d_ff, seed, dtype=torch.float64):
    """Random family member with NON-TRIVIAL norm gains (so absorption is
    genuinely exercised, not vacuously true at g=1)."""
    torch.manual_seed(seed)
    m = SAPModel(FAM, ModelWidths.uniform(FAM.n_layers, n_heads, d_ff))
    with torch.no_grad():
        for blk in m.blocks:
            blk.attn_norm.weight.uniform_(0.5, 1.5)
            blk.ffn_norm.weight.uniform_(0.5, 1.5)
        m.final_norm.weight.uniform_(0.5, 1.5)
    return m.to(dtype)


def sublayer_outputs(model, x, cos, sin):
    """Per-block (attention-sublayer, ffn-sublayer) outputs, norms included."""
    outs = []
    for blk in model.blocks:
        outs.append((
            blk.attn(blk.attn_norm(x), cos, sin),
            blk.ffn(blk.ffn_norm(x)),
        ))
    return outs


def rand_x(T=8, B=3, seed=99):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, FAM.d_model, dtype=torch.float64, generator=g)


# ---------------------------------------------------------------------------
# Core merge identities (float64 — machine-precision exact)
# ---------------------------------------------------------------------------

def test_pairwise_heterogeneous_exactness():
    """2-head+3-head attention, 6+10-neuron FFN, random gains, alpha=0.4."""
    A = make_model(n_heads=2, d_ff=6, seed=0)
    B = make_model(n_heads=3, d_ff=10, seed=1)
    M = merge_models([A, B], [0.4, 0.6])
    x = rand_x()
    cos, sin = build_rope_cache(x.shape[1], FAM.d_head, FAM.rope_theta)

    for (ma, mf), (aa, af), (ba, bf) in zip(
        sublayer_outputs(M, x, cos, sin),
        sublayer_outputs(A, x, cos, sin),
        sublayer_outputs(B, x, cos, sin),
    ):
        assert (ma - (0.4 * aa + 0.6 * ba)).abs().max().item() < TOL64
        assert (mf - (0.4 * af + 0.6 * bf)).abs().max().item() < TOL64


def test_nway_exactness():
    """3 models at once with unequal alphas (0.5, 0.3, 0.2)."""
    models = [make_model(2, 6, seed=0), make_model(3, 10, seed=1), make_model(1, 4, seed=2)]
    alphas = [0.5, 0.3, 0.2]
    M = merge_models(models, alphas)
    assert M.widths.n_heads == [6, 6] and M.widths.d_ff == [20, 20]

    x = rand_x()
    cos, sin = build_rope_cache(x.shape[1], FAM.d_head, FAM.rope_theta)
    parent_outs = [sublayer_outputs(m, x, cos, sin) for m in models]
    for li, (ma, mf) in enumerate(sublayer_outputs(M, x, cos, sin)):
        ta = sum(a * po[li][0] for a, po in zip(alphas, parent_outs))
        tf = sum(a * po[li][1] for a, po in zip(alphas, parent_outs))
        assert (ma - ta).abs().max().item() < TOL64
        assert (mf - tf).abs().max().item() < TOL64


def test_sequential_equals_simultaneous():
    """merge(merge(A,B; 1/2,1/2), C; 2/3,1/3) == merge(A,B,C; 1/3 each), weight-for-weight."""
    A, B, C = make_model(2, 6, 0), make_model(3, 10, 1), make_model(1, 4, 2)
    M_seq = merge_models([merge_models([A, B], [0.5, 0.5]), C], [2 / 3, 1 / 3])
    M_sim = merge_models([A, B, C], [1 / 3, 1 / 3, 1 / 3])
    for (ka, va), (kb, vb) in zip(
        M_seq.state_dict().items(), M_sim.state_dict().items()
    ):
        assert ka == kb
        assert torch.allclose(va, vb, atol=TOL64), f"mismatch in {ka}"


def test_overcomplete_wo():
    """6-head + 8-head with d_model=16, d_head=4: the balanced head count is 4,
    so both parents and the 14-head merge are over-complete and W_O is a
    rectangle (16 x 56). Exactness must not care."""
    A = make_model(n_heads=6, d_ff=8, seed=3)
    B = make_model(n_heads=8, d_ff=8, seed=4)
    M = merge_models([A, B], [0.5, 0.5])
    assert M.blocks[0].attn.wo.weight.shape == (16, 14 * 4)

    x = rand_x()
    cos, sin = build_rope_cache(x.shape[1], FAM.d_head, FAM.rope_theta)
    for (ma, _), (aa, _), (ba, _) in zip(
        sublayer_outputs(M, x, cos, sin),
        sublayer_outputs(A, x, cos, sin),
        sublayer_outputs(B, x, cos, sin),
    ):
        assert (ma - 0.5 * (aa + ba)).abs().max().item() < TOL64


def test_unscaled_merge_is_exact_sum():
    """scaled=False: merged sublayer == SUM of parent sublayers (not average)."""
    A = make_model(2, 6, seed=0)
    B = make_model(3, 10, seed=1)
    M = merge_models([A, B], [0.5, 0.5], scaled=False)
    x = rand_x()
    cos, sin = build_rope_cache(x.shape[1], FAM.d_head, FAM.rope_theta)
    for (ma, mf), (aa, af), (ba, bf) in zip(
        sublayer_outputs(M, x, cos, sin),
        sublayer_outputs(A, x, cos, sin),
        sublayer_outputs(B, x, cos, sin),
    ):
        assert (ma - (aa + ba)).abs().max().item() < TOL64
        assert (mf - (af + bf)).abs().max().item() < TOL64


def test_norm_absorption_preserves_function():
    """Absorbing RMSNorm gains into projections must not change the model."""
    m = make_model(3, 10, seed=5)
    idx = torch.randint(0, FAM.vocab_size, (2, 12))
    y0, _ = m(idx)
    absorb_norm_gains(m)
    for blk in m.blocks:  # gains really were reset
        assert torch.all(blk.attn_norm.weight == 1.0)
        assert torch.all(blk.ffn_norm.weight == 1.0)
    y1, _ = m(idx)
    assert (y0 - y1).abs().max().item() < 1e-10


# ---------------------------------------------------------------------------
# The gate itself: passes correct merges, catches corrupted ones
# ---------------------------------------------------------------------------

def test_verify_merge_passes_and_reports():
    A, B = make_model(2, 6, 0), make_model(3, 10, 1)
    M = merge_models([A, B], [0.4, 0.6])
    report = verify_merge([A, B], M, [0.4, 0.6], scaled=True, tol=1e-10)
    assert report["max_err"] < 1e-10


def test_verify_merge_catches_corruption():
    """A deliberately wrong merge (wrong scale on one wo slice) must FAIL the
    gate — proof the check is not a rubber stamp."""
    A, B = make_model(2, 6, 0), make_model(3, 10, 1)
    M = merge_models([A, B], [0.4, 0.6])
    with torch.no_grad():
        M.blocks[1].attn.wo.weight[:, :8].mul_(2.0)   # corrupt A's head slice
    with pytest.raises(MergeExactnessError):
        verify_merge([A, B], M, [0.4, 0.6], scaled=True, tol=1e-6)


def test_verify_merge_unscaled_mode():
    A, B = make_model(2, 6, 0), make_model(3, 10, 1)
    M = merge_models([A, B], [0.5, 0.5], scaled=False)
    report = verify_merge([A, B], M, [0.5, 0.5], scaled=False, tol=1e-10)
    assert report["max_err"] < 1e-10
    # and the scaled expectation must NOT hold for an unscaled merge
    with pytest.raises(MergeExactnessError):
        verify_merge([A, B], M, [0.5, 0.5], scaled=True, tol=1e-6)


# ---------------------------------------------------------------------------
# Baseline + alpha bookkeeping
# ---------------------------------------------------------------------------

def test_weight_average_requires_identical_widths():
    A, B = make_model(2, 6, 0), make_model(3, 10, 1)
    with pytest.raises(ValueError):
        weight_average([A, B], [0.5, 0.5])
    C, D = make_model(2, 6, 6), make_model(2, 6, 7)
    avg = weight_average([C, D], [0.5, 0.5])
    w = avg.blocks[0].attn.wq.weight
    expect = 0.5 * (C.blocks[0].attn.wq.weight + D.blocks[0].attn.wq.weight)
    assert torch.allclose(w, expect, atol=TOL64)


def test_alpha_resolution():
    assert resolve_alphas(2, "tokens", token_counts=[300, 100]) == [0.75, 0.25]
    assert resolve_alphas(4, "uniform") == [0.25] * 4
    assert resolve_alphas(2, "manual", manual_alphas=[7, 3]) == [0.7, 0.3]
    with pytest.raises(ValueError):
        resolve_alphas(2, "tokens", token_counts=[100, 0])


# ---------------------------------------------------------------------------
# Growth law — exact to the parameter
# ---------------------------------------------------------------------------

def test_parameter_growth_accounting():
    """S_merged = E + sum(S_i - E) up to the norm gains, which are counted
    once instead of N times: exact correction is (N-1)*(2L+1)*d_model."""
    models = [make_model(2, 6, 0), make_model(3, 10, 1), make_model(1, 4, 2)]
    M = merge_models(models, [1 / 3, 1 / 3, 1 / 3])
    E = models[0].param_counts()["embedding"]
    norm_params = (2 * FAM.n_layers + 1) * FAM.d_model
    expected = (
        E
        + sum(m.param_counts()["total"] - E for m in models)
        - (len(models) - 1) * norm_params
    )
    assert M.param_counts()["total"] == expected


# ---------------------------------------------------------------------------
# Checkpoints, growth, and crash-resume
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip(tmp_path):
    m = make_model(3, 10, seed=8, dtype=torch.float32)
    meta = {"role": "shard", "name": "t", "tokens_seen": 12345}
    p = tmp_path / "m.pt"
    save_model_checkpoint(p, m, meta)
    m2, fam2, w2, meta2 = load_model(p)
    assert fam2.to_dict() == FAM.to_dict()
    assert meta2["tokens_seen"] == 12345
    assert w2.to_dict() == m.widths.to_dict()
    idx = torch.randint(0, FAM.vocab_size, (2, 12))
    y1, _ = m(idx)
    y2, _ = m2(idx)
    assert torch.equal(y1, y2)


def test_widths_derived_from_weights():
    m = make_model(5, 12, seed=9, dtype=torch.float32)
    derived = widths_from_state_dict(m.state_dict(), FAM)
    assert derived.n_heads == [5, 5] and derived.d_ff == [12, 12]


def test_grow_from_seed_is_function_preserving(tmp_path):
    """Branching into a WIDER shard must reproduce the seed's function
    exactly at branch time (new units have zero output weights)."""
    from sap.train import grow_from_checkpoint
    torch.manual_seed(11)
    seed_model = SAPModel(FAM, ModelWidths.uniform(FAM.n_layers, 2, 6))
    p = tmp_path / "seed.pt"
    save_model_checkpoint(p, seed_model, {"role": "seed", "name": "seed", "tokens_seen": 10})

    target = ModelWidths.uniform(FAM.n_layers, 4, 12)
    grown, src_meta = grow_from_checkpoint(str(p), FAM, target, init_seed=99)
    assert src_meta["tokens_seen"] == 10
    assert grown.widths.n_heads == [4, 4]

    idx = torch.randint(0, FAM.vocab_size, (2, 16))
    y_seed, _ = seed_model(idx)
    y_grown, _ = grown(idx)
    assert (y_seed - y_grown).abs().max().item() < 1e-5   # float32 tolerance


def test_chunk_sampler_resume_and_coverage():
    s1 = ChunkSampler(n_tokens=10_001, seq_len=100, batch_size=7, data_seed=5)
    first = [s1.next_batch().copy() for _ in range(10)]
    state = s1.state()
    rest = [s1.next_batch().copy() for _ in range(10)]
    s2 = ChunkSampler(n_tokens=10_001, seq_len=100, batch_size=7, data_seed=5,
                      epoch=state["epoch"], cursor=state["cursor"])
    rest2 = [s2.next_batch().copy() for _ in range(10)]
    for a, b in zip(rest, rest2):
        assert np.array_equal(a, b)
    # one epoch covers every chunk at most once, no repeats
    s3 = ChunkSampler(n_tokens=10_001, seq_len=100, batch_size=7, data_seed=5)
    seen = np.concatenate([s3.next_batch() for _ in range(s3.batches_per_epoch)])
    assert len(seen) == len(set(seen.tolist()))


def _write_tiny_dataset(dirpath: Path, n_tokens=30_000, vocab=64, seed=0):
    rng = np.random.RandomState(seed)
    toks = rng.randint(0, vocab, size=n_tokens).astype(np.uint16)
    dirpath.mkdir(parents=True, exist_ok=True)
    toks.tofile(dirpath / "data.bin")
    with open(dirpath / "meta.json", "w") as f:
        json.dump({"dtype": "uint16", "vocab_size": vocab}, f)
    return dirpath / "data.bin"


def _tiny_cfg(data_bin, out_dir, **overrides):
    from sap.train import TrainConfig
    base = dict(
        name="t", out_dir=str(out_dir), data_path=str(data_bin),
        family=FAM, n_heads_spec="2", d_ff_spec="8",
        batch_size=4, grad_accum=1, seq_len=16,
        lr=1e-3, min_lr=1e-4, warmup_steps=2, schedule="cosine",
        max_steps=8, checkpoint_every_min=None, checkpoint_every_steps=None,
        eval_interval=None, log_interval=1000,
        device="cpu", dtype="fp32", init_seed=7, data_seed=3,
    )
    base.update(overrides)
    return TrainConfig(**base)


def test_crash_resume_reproduces_uninterrupted_run(tmp_path):
    """Train 8 steps straight vs. train 4 steps, 'crash', resume to 8 —
    the final weights must match. This is the checkpoint system's whole job.

    Uses the constant schedule: cosine LR is a function of the configured
    horizon (max_steps), so extending the budget mid-run legitimately changes
    future LRs — that would test the schedule, not the resume machinery."""
    from sap.train import run_training
    data_bin = _write_tiny_dataset(tmp_path / "data")

    run_training(_tiny_cfg(data_bin, tmp_path / "straight", max_steps=8,
                           schedule="constant"))
    run_training(_tiny_cfg(data_bin, tmp_path / "resumed", max_steps=4,
                           schedule="constant"))
    # simulated crash + rerun with extended budget (same everything else)
    summary = run_training(_tiny_cfg(data_bin, tmp_path / "resumed", max_steps=8,
                                     schedule="constant"))
    assert summary["steps"] == 8

    m1, _, _, meta1 = load_model(tmp_path / "straight" / "final.pt")
    m2, _, _, meta2 = load_model(tmp_path / "resumed" / "final.pt")
    assert meta1["tokens_seen"] == meta2["tokens_seen"]
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert k1 == k2
        assert torch.allclose(v1, v2, atol=1e-6), f"weights diverged after resume: {k1}"


def test_checkpoint_rotation_keeps_prev_generations(tmp_path):
    """latest.pt plus at most keep_prev previous generations, rotated by
    renames; every retained file must be loadable."""
    from sap.train import run_training
    data_bin = _write_tiny_dataset(tmp_path / "data")
    run_training(_tiny_cfg(data_bin, tmp_path / "run", max_steps=8,
                           checkpoint_every_steps=2, schedule="constant",
                           keep_prev=2))
    d = tmp_path / "run"
    assert (d / "latest.pt").exists() and (d / "final.pt").exists()
    assert (d / "prev_1.pt").exists() and (d / "prev_2.pt").exists()
    assert not (d / "prev_3.pt").exists()          # capped at keep_prev
    assert not (d / "latest.pt.new").exists()      # no leftover temp
    for f in ("latest.pt", "prev_1.pt", "prev_2.pt"):
        m, _, _, _ = load_model(d / f)             # all generations loadable


def test_resume_falls_back_to_prev_when_latest_lost(tmp_path):
    """Simulate a crash in the rotation window: latest.pt gone, prev_1 intact.
    The rerun must resume from prev_1 and still complete the extended budget."""
    from sap.train import run_training
    data_bin = _write_tiny_dataset(tmp_path / "data")
    run_training(_tiny_cfg(data_bin, tmp_path / "run", max_steps=4,
                           checkpoint_every_steps=2, schedule="constant"))
    (tmp_path / "run" / "latest.pt").unlink()
    s = run_training(_tiny_cfg(data_bin, tmp_path / "run", max_steps=8,
                               checkpoint_every_steps=2, schedule="constant"))
    assert s["steps"] == 8 and s["stop_reason"] == "max_steps"


def test_resume_promotes_interrupted_new_checkpoint(tmp_path):
    """Simulate a crash between rotation and the final rename: only
    latest.pt.new exists. It must be promoted and used."""
    from sap.train import run_training
    data_bin = _write_tiny_dataset(tmp_path / "data")
    run_training(_tiny_cfg(data_bin, tmp_path / "run", max_steps=4,
                           schedule="constant"))
    d = tmp_path / "run"
    (d / "latest.pt").rename(d / "latest.pt.new")
    s = run_training(_tiny_cfg(data_bin, d, max_steps=8, schedule="constant"))
    assert s["steps"] == 8
    assert (d / "latest.pt").exists() and not (d / "latest.pt.new").exists()


def test_extend_epochs_continues_a_finished_run(tmp_path):
    """'Keep improving after the budget': a completed 1-epoch shard extended
    by --extend-epochs 1 trains exactly one more epoch."""
    from sap.train import run_training
    data_bin = _write_tiny_dataset(tmp_path / "data", n_tokens=2_000)
    s1 = run_training(_tiny_cfg(data_bin, tmp_path / "run", max_steps=None,
                                max_epochs=1, schedule="constant"))
    assert s1["stop_reason"] == "max_epochs" and s1["epochs_completed"] == 1
    s2 = run_training(_tiny_cfg(data_bin, tmp_path / "run", max_steps=None,
                                extend_epochs=1, schedule="constant"))
    assert s2["stop_reason"] == "max_epochs" and s2["epochs_completed"] == 2
    assert s2["tokens_seen"] == 2 * s1["tokens_seen"]


def test_status_file_reports_completion(tmp_path):
    from sap.train import run_training
    data_bin = _write_tiny_dataset(tmp_path / "data")
    run_training(_tiny_cfg(data_bin, tmp_path / "run", max_steps=6,
                           schedule="constant"))
    status = json.load(open(tmp_path / "run" / "status.json"))
    assert status["state"] == "completed"
    assert status["stop_reason"] == "max_steps"
    assert status["step"] == 6 and status["progress"] == 1.0


def test_trainer_refuses_out_of_range_token_ids(tmp_path):
    """A .bin containing an id >= vocab_size must be REFUSED with a clear
    error at startup — never allowed to reach the embedding lookup, where it
    would index out of bounds inside native code (the one mechanism by which
    corrupt data could hard-crash training)."""
    from sap.train import run_training
    d = tmp_path / "data"
    d.mkdir(parents=True)
    toks = np.random.RandomState(0).randint(0, 64, size=5_000).astype(np.uint16)
    toks[1234] = 64  # vocab_size is 64 -> valid ids are 0..63; this one is OOB
    toks.tofile(d / "data.bin")
    with open(d / "meta.json", "w") as f:
        json.dump({"dtype": "uint16", "vocab_size": 64}, f)
    with pytest.raises(ValueError, match="vocab_size"):
        run_training(_tiny_cfg(d / "data.bin", tmp_path / "run", max_steps=2))


def test_resume_rejects_changed_critical_args(tmp_path):
    from sap.train import run_training
    data_bin = _write_tiny_dataset(tmp_path / "data")
    run_training(_tiny_cfg(data_bin, tmp_path / "run", max_steps=2))
    with pytest.raises(ValueError, match="resume mismatch"):
        run_training(_tiny_cfg(data_bin, tmp_path / "run", max_steps=4, batch_size=8))


def test_epoch_budget_stops_training(tmp_path):
    from sap.train import run_training
    data_bin = _write_tiny_dataset(tmp_path / "data", n_tokens=2_000)
    cfg = _tiny_cfg(data_bin, tmp_path / "run", max_steps=None, max_epochs=2)
    summary = run_training(cfg)
    assert summary["stop_reason"] == "max_epochs"
    assert summary["epochs_completed"] == 2


# ---------------------------------------------------------------------------
# End-to-end merge of TRAINED (not random) models through the file interface
# ---------------------------------------------------------------------------

def test_end_to_end_train_merge_verify(tmp_path):
    """Two tiny shards trained on different data, merged via merge_checkpoints
    (token-weighted alphas), gate green, lineage bookkeeping correct."""
    from sap.merge import merge_checkpoints
    from sap.train import run_training

    binA = _write_tiny_dataset(tmp_path / "dA", seed=1)
    binB = _write_tiny_dataset(tmp_path / "dB", seed=2)
    run_training(_tiny_cfg(binA, tmp_path / "sA", max_steps=6))
    run_training(_tiny_cfg(binB, tmp_path / "sB", max_steps=12,
                           n_heads_spec="3", d_ff_spec="10", init_seed=8))

    out = tmp_path / "merged.pt"
    meta, report = merge_checkpoints(
        [tmp_path / "sA" / "final.pt", tmp_path / "sB" / "final.pt"],
        out, alpha_mode="tokens", scaled=True, check=True, tol=1e-3,
    )
    assert report["max_err"] < 1e-3
    a = [e["alpha"] for e in meta["lineage"]]
    assert abs(a[0] - 6 / 18) < 1e-9 and abs(a[1] - 12 / 18) < 1e-9   # 6:12 steps => 1:2 tokens
    m, fam, w, mmeta = load_model(out)
    assert w.n_heads == [5, 5] and w.d_ff == [18, 18]
    assert mmeta["tokens_seen"] == meta["tokens_seen"]
