#!/usr/bin/env python
"""Hardware / driver-stack triage for the SAP training box.

Repeated SIGSEGVs across UNRELATED programs (CPU-only tokenization AND GPU
training) plus full-system crashes cannot be caused by Python-level code —
they point below: host RAM, GPU/driver, or power delivery. This script tests
each layer in isolation and tells you which one misbehaves.

Run it on the research PC (takes ~3-5 minutes by default):

    python diagnose.py                # default: ~2 min RAM + ~2 min GPU
    python diagnose.py --ram-gb 24 --minutes 10    # longer, more sensitive

How to read the outcome:
  * crashes/mismatch in PHASE 2 (CPU RAM)  -> bad host RAM (run memtest86 to confirm)
  * crashes/mismatch in PHASE 3+ (GPU)     -> GPU / driver / power problem
  * everything clean but training still dies -> run `dmesg -T | tail -50`
    immediately after the next crash and read the last lines (Xid = GPU driver
    error, 'Out of memory' = OOM-killer, MCE = CPU/RAM fault)

Each phase prints BEFORE it runs, so if the script itself dies you still know
exactly which layer killed it — that IS the diagnosis.
"""

import argparse
import sys
import time
import zlib

import numpy as np


def banner(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def phase_versions() -> bool:
    banner("PHASE 1: environment")
    import torch
    print(f"python  : {sys.version.split()[0]}")
    print(f"torch   : {torch.__version__} (built for CUDA {torch.version.cuda})")
    has_cuda = torch.cuda.is_available()
    print(f"cuda    : available={has_cuda}", flush=True)
    if has_cuda:
        print(f"device  : {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"vram    : {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total", flush=True)
    return has_cuda


def phase_cpu_ram(ram_gb: float, seconds: float) -> bool:
    """Write deterministic patterns into a large host buffer, checksum, wait,
    re-checksum. Any difference = a bit flipped in RAM = faulty memory."""
    banner(f"PHASE 2: host RAM pattern test (~{ram_gb:.0f} GB, {seconds:.0f}s)")
    block_bytes = 512 * 1024 * 1024
    n_blocks = max(1, int(ram_gb * 1e9 // block_bytes))
    blocks, sums = [], []
    print(f"allocating {n_blocks} x 512MB and writing patterns...", flush=True)
    for i in range(n_blocks):
        a = np.empty(block_bytes, dtype=np.uint8)
        a[:] = np.arange(block_bytes, dtype=np.uint64).astype(np.uint8)  # pattern
        a[:: 4096] = (i * 37 + 11) % 251                                  # per-block salt
        blocks.append(a)
        sums.append(zlib.crc32(a))
    print("re-verifying checksums in passes...", flush=True)
    t_end = time.time() + seconds
    passes, bad = 0, 0
    while time.time() < t_end:
        for i, a in enumerate(blocks):
            if zlib.crc32(a) != sums[i]:
                bad += 1
                print(f"  *** BIT FLIP detected in block {i} on pass {passes} ***",
                      flush=True)
        passes += 1
    del blocks
    print(f"completed {passes} verification passes: "
          + ("NO corruption — host RAM looks fine at this load"
         if bad == 0 else f"*** {bad} CORRUPTED READS — HOST RAM IS FAULTY ***"))
    return bad == 0


def _gpu_repeat(name: str, fn, iters: int) -> bool:
    """Run fn() repeatedly; identical inputs through the identical kernel must
    produce bitwise-identical outputs. Any deviation = unstable GPU compute."""
    import torch
    print(f"  {name}: ", end="", flush=True)
    ref = fn()
    torch.cuda.synchronize()
    for _ in range(iters):
        out = fn()
        torch.cuda.synchronize()
        if not torch.equal(ref, out):
            print("*** NON-DETERMINISTIC RESULT — GPU compute unstable ***")
            return False
    print(f"OK ({iters} identical repeats)")
    return True


def phase_gpu(seconds: float) -> bool:
    banner(f"PHASE 3: GPU compute stability (~{seconds:.0f}s sustained load)")
    import torch
    import torch.nn.functional as F
    dev = "cuda"
    ok = True

    g = torch.Generator(device=dev).manual_seed(0)
    A = torch.randn(4096, 4096, device=dev, generator=g)
    B = torch.randn(4096, 4096, device=dev, generator=g)
    Ah, Bh = A.bfloat16(), B.bfloat16()
    q = torch.randn(8, 8, 1024, 64, device=dev, dtype=torch.bfloat16, generator=g)

    per = max(20, int(seconds * 12))  # rough iteration count per sub-test
    ok &= _gpu_repeat("fp32 matmul     ", lambda: A @ B, per)
    ok &= _gpu_repeat("bf16 matmul     ", lambda: Ah @ Bh, per)
    ok &= _gpu_repeat("sdpa MATH kernel",
                      lambda: F.scaled_dot_product_attention(q, q, q, is_causal=True),
                      per // 2)
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            ok &= _gpu_repeat("sdpa FLASH kernel",
                              lambda: F.scaled_dot_product_attention(q, q, q, is_causal=True),
                              per // 2)
    except Exception as e:  # noqa: BLE001
        print(f"  sdpa FLASH kernel: could not run in isolation ({e}); skipped")

    # sustained mixed load (the closest to a real training step)
    print("  sustained train-like load: ", end="", flush=True)
    t_end = time.time() + seconds / 2
    n = 0
    while time.time() < t_end:
        (Ah @ Bh).float().sum().item()
        F.scaled_dot_product_attention(q, q, q, is_causal=True).sum().item()
        n += 1
    print(f"OK ({n} rounds, no crash)")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ram-gb", type=float, default=8.0,
                    help="host RAM to pattern-test (leave headroom for the OS!)")
    ap.add_argument("--minutes", type=float, default=4.0,
                    help="total test time, split across phases")
    args = ap.parse_args()

    half = args.minutes * 60 / 2
    has_cuda = phase_versions()
    ram_ok = phase_cpu_ram(args.ram_gb, half)
    gpu_ok = phase_gpu(half) if has_cuda else False

    banner("VERDICT")
    if not ram_ok:
        print("HOST RAM IS FAULTY. This explains segfaults in unrelated programs\n"
              "(tokenizer AND training) and full-system crashes. Confirm with a\n"
              "memtest86 boot pass and replace/reseat the RAM. No software change\n"
              "can fix this.")
    elif has_cuda and not gpu_ok:
        print("GPU COMPUTE IS UNSTABLE (non-deterministic results under load).\n"
              "Suspects in order: power delivery (transient spikes tripping the\n"
              "PSU), GPU driver install, GPU hardware. A power cap "
              "(`nvidia-smi -pl 300`,\nmanual + non-persistent) is the quickest "
              "experiment to separate power\nfrom silicon.")
    elif not has_cuda:
        print("CUDA is not available — fix the driver before anything else.")
    else:
        print("All phases CLEAN at this load. If training still crashes, capture\n"
              "the kernel's view immediately after the next crash:\n"
              "    dmesg -T | tail -50\n"
              "and look for: 'NVRM: Xid' (GPU driver fault), 'Out of memory'\n"
              "(OOM killer), or 'mce:' (CPU/RAM machine-check error). That line\n"
              "names the guilty component directly.")


if __name__ == "__main__":
    main()
