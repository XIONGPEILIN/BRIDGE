#!/usr/bin/env python3
"""Precompute the empty-prompt ("") Qwen3 embedding = the CFG *unconditional / null* prompt.

This is the SAME thing the Klein-base inference pipeline uses as its CFG negative branch
(`negative_prompt=""` -> `_get_qwen3_prompt_embeds("")`), so prompt-dropout CFG *training* and
two-pass text-CFG *inference* use an identical unconditional embedding (no train/inference skew).

Output: `<cache_dir>/_null_prompt_embeds.safetensors` with `null_prompt_embeds` [1, S, D] and
`null_text_ids` [1, S, 4]. Run once; the training script loads it for `--prompt_drop_prob`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_DIFFUSERS_SRC = REPO_ROOT / "diffusers" / "src"
if LOCAL_DIFFUSERS_SRC.exists() and str(LOCAL_DIFFUSERS_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIFFUSERS_SRC))

from diffusers import Flux2KleinPipeline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained_model_name_or_path", default="black-forest-labs/FLUX.2-klein-base-9B")
    p.add_argument("--cache_dir", type=Path, default=REPO_ROOT / "cache_full")
    p.add_argument("--text_encoder_out_layers", default="9,18,27")
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--negative_prompt", default="",
                   help="The unconditional prompt to encode. Klein-base inference uses '' (empty).")
    args = p.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    layers = tuple(int(x.strip()) for x in args.text_encoder_out_layers.split(",") if x.strip())
    device = torch.device(args.device)

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", local_files_only=args.local_files_only,
    )
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        torch_dtype=dtype, local_files_only=args.local_files_only,
    ).to(device).eval()

    with torch.no_grad():
        null_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=args.negative_prompt,
            device=device,
            dtype=dtype,
            max_sequence_length=args.max_sequence_length,
            hidden_states_layers=layers,
        )
        null_text_ids = Flux2KleinPipeline._prepare_text_ids(null_embeds).to(device)

    out = args.cache_dir / "_null_prompt_embeds.safetensors"
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {"null_prompt_embeds": null_embeds.cpu().contiguous(),
         "null_text_ids": null_text_ids.cpu().contiguous()},
        str(out),
        metadata={"negative_prompt": args.negative_prompt,
                  "layers": args.text_encoder_out_layers,
                  "max_sequence_length": str(args.max_sequence_length)},
    )
    print(f"null_prompt_embeds {tuple(null_embeds.shape)} {null_embeds.dtype}")
    print(f"null_text_ids      {tuple(null_text_ids.shape)} {null_text_ids.dtype}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
