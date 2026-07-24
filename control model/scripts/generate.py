#!/usr/bin/env python
"""Interactive text generation from any checkpoint — the manual smell test.

Loads a shard, a merged model, the conventional baseline, or the backbone of a
fine-tuned classifier, and generates continuations from prompts you type. This
is the qualitative counterpart to perplexity: numbers tell you the model is
better, reading its output tells you *how*.

Sampling is implemented here rather than calling SAPModel.generate so that
top-p and a repetition penalty are available; the model code stays untouched.

Examples
--------
# interactive session (type prompts, Ctrl-C or /quit to leave)
python scripts/generate.py --model runs/merged/merged.pt

# one-shot
python scripts/generate.py --model runs/baseline_546m/final.pt \
    --prompt "The three laws of thermodynamics state that" --max-new-tokens 120

# greedy, fully deterministic
python scripts/generate.py --model runs/merged/merged.pt \
    --prompt "In 1969, the Apollo 11 mission" --greedy

# compare two models on the same prompts, side by side
python scripts/generate.py --model runs/merged/merged.pt \
    --compare runs/baseline_546m/final.pt \
    --prompt "Photosynthesis is the process by which" --seed 0

Interactive commands
--------------------
  /quit /exit        leave
  /temp 0.7          set temperature
  /topk 50           set top-k        (0 disables)
  /topp 0.9          set top-p        (1.0 disables)
  /rep 1.1           set repetition penalty (1.0 disables)
  /len 200           set max new tokens
  /seed 123          reseed the sampler
  /greedy            toggle greedy decoding
  /params            show current settings
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sap.config import FamilyConfig, ModelWidths  # noqa: E402
from sap.model import SAPModel, load_checkpoint, model_from_checkpoint  # noqa: E402


# ---------------------------------------------------------------------------
# Loading: accepts a plain family checkpoint OR a fine-tuned classifier
# ---------------------------------------------------------------------------

def load_any(path: str, device: torch.device):
    """Return (model, label). Fine-tuned checkpoints contribute their backbone."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("kind") == "sap_finetune_classifier":
        family = FamilyConfig.from_dict(ck["family"])
        widths = ModelWidths.from_dict(ck["widths"])
        model = SAPModel(family, widths)
        model.load_state_dict(ck["backbone_state"])
        model.to(device).eval()
        meta = ck.get("meta", {})
        label = (f"{meta.get('name', Path(path).stem)} "
                 f"[fine-tuned backbone, dataset={meta.get('dataset', '?')}]")
        return model, label
    if "format_version" not in ck:
        raise ValueError(f"{path} is not a SAP checkpoint")
    model, family, widths, meta = model_from_checkpoint(ck, device=device)
    model.eval()
    counts = model.param_counts()
    label = (f"{meta.get('name', Path(path).stem)} "
             f"[{meta.get('role', '?')}, {counts['total'] / 1e6:.1f}M params]")
    return model, label


class Decoder:
    """Family tokenizer for prompt encoding and output decoding."""

    def __init__(self, name: str = "gpt2", backend: str = "tiktoken"):
        self.backend = backend
        if backend == "tiktoken":
            import tiktoken
            self.enc = tiktoken.get_encoding(name)
            self.eot = self.enc.eot_token
        else:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from transformers import AutoTokenizer
            self.enc = AutoTokenizer.from_pretrained(name)
            self.eot = self.enc.eos_token_id

    def encode(self, text: str) -> List[int]:
        if self.backend == "tiktoken":
            return self.enc.encode_ordinary(text)
        return self.enc(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids: List[int]) -> str:
        if self.backend == "tiktoken":
            return self.enc.decode(ids)
        return self.enc.decode(ids, skip_special_tokens=False)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: SAPModel,
    dec: Decoder,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    greedy: bool = False,
    stop_at_eot: bool = True,
    stream: bool = True,
    device: torch.device = torch.device("cpu"),
) -> str:
    ids = dec.encode(prompt)
    if not ids:
        raise ValueError("prompt tokenized to nothing")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    n_prompt = idx.shape[1]
    max_ctx = model.family.max_seq_len

    printed = ""
    t0 = time.monotonic()
    generated = 0
    for _ in range(max_new_tokens):
        ctx = idx[:, -max_ctx:]
        logits, _ = model(ctx)
        logits = logits[:, -1, :].float()

        if repetition_penalty != 1.0:
            # penalize tokens already present: divide positive logits, multiply
            # negative ones (the CTRL formulation), so the effect is monotone
            seen = torch.unique(idx)
            vals = logits[0, seen]
            logits[0, seen] = torch.where(vals > 0, vals / repetition_penalty,
                                          vals * repetition_penalty)

        if greedy or temperature <= 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k and top_k > 0:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p and top_p < 1.0:
                srt, si = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(srt, dim=-1)
                cum = probs.cumsum(dim=-1)
                drop = cum - probs > top_p          # keep the token that crosses p
                srt = srt.masked_fill(drop, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(1, si, srt)
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

        idx = torch.cat([idx, nxt], dim=1)
        generated += 1

        if stop_at_eot and int(nxt.item()) == dec.eot:
            break
        if stream:
            full = dec.decode(idx[0, n_prompt:].tolist())
            sys.stdout.write(full[len(printed):])
            sys.stdout.flush()
            printed = full

    text = dec.decode(idx[0, n_prompt:].tolist())
    if stream:
        sys.stdout.write(text[len(printed):])
        sys.stdout.flush()
        dt = time.monotonic() - t0
        rate = generated / dt if dt > 0 else 0.0
        sys.stdout.write(f"\n\n  [{generated} tokens in {dt:.1f}s — {rate:.1f} tok/s]\n")
        sys.stdout.flush()
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="checkpoint to generate from")
    p.add_argument("--compare", nargs="*", default=[],
                   help="additional checkpoints to run on the same prompt")
    p.add_argument("--prompt", default=None,
                   help="one-shot prompt; omit for an interactive session")
    p.add_argument("--prompt-file", default=None,
                   help="text file with one prompt per line (batch mode)")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50, help="0 disables")
    p.add_argument("--top-p", type=float, default=1.0, help="1.0 disables")
    p.add_argument("--repetition-penalty", type=float, default=1.0, help="1.0 disables")
    p.add_argument("--greedy", action="store_true", help="deterministic argmax decoding")
    p.add_argument("--no-stop-at-eot", action="store_true",
                   help="keep generating past the end-of-text token")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--tokenizer-backend", choices=["tiktoken", "hf"], default="tiktoken")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    dec = Decoder(args.tokenizer, args.tokenizer_backend)

    paths = [args.model] + list(args.compare)
    models = []
    for path in paths:
        if not Path(path).exists():
            p.error(f"checkpoint not found: {path}")
        m, label = load_any(path, device)
        models.append((label, m))
        print(f"loaded: {label}")
    print(f"device: {device}")
    if device.type == "cpu":
        print("NOTE: generating on CPU. There is no KV cache, so every new token "
              "re-runs the full forward pass — expect this to be slow at scale.")

    state = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
        "greedy": args.greedy,
    }

    def run(prompt: str) -> None:
        if args.seed is not None:
            torch.manual_seed(args.seed)
        for label, m in models:
            if len(models) > 1:
                print(f"\n----- {label} -----")
            print(f"\n\033[2m{prompt}\033[0m", end="")
            generate(
                m, dec, prompt,
                max_new_tokens=state["max_new_tokens"],
                temperature=state["temperature"],
                top_k=state["top_k"],
                top_p=state["top_p"],
                repetition_penalty=state["repetition_penalty"],
                greedy=state["greedy"],
                stop_at_eot=not args.no_stop_at_eot,
                stream=True,
                device=device,
            )
            if args.seed is not None:
                torch.manual_seed(args.seed)   # same seed for every model

    # -- batch from file ------------------------------------------------------
    if args.prompt_file:
        for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                print("\n" + "=" * 70)
                run(line)
        return

    # -- one shot -------------------------------------------------------------
    if args.prompt is not None:
        run(args.prompt)
        return

    # -- interactive ----------------------------------------------------------
    print("\nInteractive mode. Type a prompt and press Enter. "
          "/help for commands, /quit to exit.\n")
    while True:
        try:
            line = input("\n\033[1mprompt>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return
        if not line:
            continue
        if line.startswith("/"):
            parts = line.split()
            cmd = parts[0].lower()
            val = parts[1] if len(parts) > 1 else None
            try:
                if cmd in ("/quit", "/exit"):
                    print("bye.")
                    return
                elif cmd == "/help":
                    print(__doc__.split("Interactive commands")[1])
                elif cmd == "/temp":
                    state["temperature"] = float(val)
                elif cmd == "/topk":
                    state["top_k"] = int(val)
                elif cmd == "/topp":
                    state["top_p"] = float(val)
                elif cmd == "/rep":
                    state["repetition_penalty"] = float(val)
                elif cmd == "/len":
                    state["max_new_tokens"] = int(val)
                elif cmd == "/seed":
                    args.seed = int(val)
                    print(f"seed = {args.seed}")
                elif cmd == "/greedy":
                    state["greedy"] = not state["greedy"]
                    print(f"greedy = {state['greedy']}")
                elif cmd == "/params":
                    for k, v in state.items():
                        print(f"  {k:<20} {v}")
                    print(f"  {'seed':<20} {args.seed}")
                else:
                    print(f"unknown command {cmd} — /help for the list")
                    continue
                if cmd in ("/temp", "/topk", "/topp", "/rep", "/len"):
                    print(f"{cmd[1:]} = {val}")
            except (TypeError, ValueError):
                print(f"bad argument for {cmd}")
            continue
        run(line)


if __name__ == "__main__":
    main()
