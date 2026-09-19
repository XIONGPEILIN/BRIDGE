#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import safetensors.torch
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_DIFFUSERS_SRC = REPO_ROOT / "diffusers" / "src"
if LOCAL_DIFFUSERS_SRC.exists() and str(LOCAL_DIFFUSERS_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIFFUSERS_SRC))

from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline  # noqa: E402

from qwen_pe_exchange_sparse_model import build_sparse_token_selection_from_mask  # noqa: E402


DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-base-9B"
DEFAULT_DATASET = REPO_ROOT / "dataset_qwen_subject_driven_best_of_three_mapped_abs.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "cache_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache samples from a subject-driven dataset.")
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pretrained_model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--text-encoder-out-layers", default="9,18,27")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true", help="Skip already cached samples")
    parser.add_argument("--gc-every", type=int, default=100, help="gc.collect + empty_cache every N samples")
    parser.add_argument("--start-index", type=int, default=None, help="Start index (inclusive)")
    parser.add_argument("--end-index", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--smoke", action="store_true", help="Use smoke test JSON format (records key, relative paths)")
    return parser.parse_args()


def to_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float32


def image_to_tensor(path: Path, *, width: int, height: int) -> torch.Tensor:
    """Stretch-resize an image to an exact (width, height) and normalize to [-1, 1].

    Used for the subject crop, whose target size is dictated by the mask crop bounds.
    """
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    return pil_to_tensor(image)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert an RGB PIL image to a normalized [-1, 1] tensor of shape (1, 3, H, W)."""
    width, height = image.size
    tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).view(height, width, 3)
    tensor = tensor.permute(2, 0, 1).to(torch.float32).div_(255.0)
    return tensor.mul_(2.0).sub_(1.0).unsqueeze(0).contiguous()


def _resize_and_crop(image: Image.Image, width: int, height: int, resample) -> Image.Image:
    """Aspect-preserving resize-to-cover then center-crop.

    Mirrors ``VaeImageProcessor._resize_and_crop`` (resize_mode="crop") used by the
    Flux2 Klein pipeline for condition images.
    """
    ratio = width / height
    src_ratio = image.width / image.height
    src_w = width if ratio > src_ratio else image.width * height // image.height
    src_h = height if ratio <= src_ratio else image.height * width // image.width
    resized = image.resize((src_w, src_h), resample=resample)
    res = Image.new(image.mode, (width, height))
    res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    return res


def aligned_size(width: int, height: int, *, multiple_of: int, max_area: int) -> tuple[int, int]:
    """Compute the Flux2 Klein target (width, height) for a source of size (width, height).

    Matches ``Flux2KleinPipeline.__call__`` image preprocessing: if the area exceeds
    ``max_area`` the size is scaled down preserving aspect ratio, then both dimensions
    are floored to a multiple of ``multiple_of`` (= vae_scale_factor * 2 = 16).
    """
    if width * height > max_area:
        scale = math.sqrt(max_area / (width * height))
        width = max(1, int(width * scale))
        height = max(1, int(height * scale))
    tw = max(multiple_of, (width // multiple_of) * multiple_of)
    th = max(multiple_of, (height // multiple_of) * multiple_of)
    return tw, th


def preprocess_aligned_image(
    path: Path,
    *,
    multiple_of: int,
    max_area: int,
    force_size: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, int, int]:
    """Load an image and resize it to a 16-aligned resolution (no stretching).

    Replicates the Flux2 Klein condition-image preprocessing: optional area cap +
    multiple-of-16 alignment + aspect-preserving resize/center-crop (LANCZOS).
    When ``force_size`` is given, the image is aligned to exactly that (w, h) so that
    images sharing the same scene (target / background / mask) land on the same grid.
    """
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    if force_size is not None:
        tw, th = force_size
    else:
        tw, th = aligned_size(*image.size, multiple_of=multiple_of, max_area=max_area)
    image = _resize_and_crop(image, tw, th, Image.Resampling.LANCZOS)
    return pil_to_tensor(image), tw, th


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    data = json.loads(args.dataset_json.read_text())
    if args.smoke:
        records = data["records"]
    else:
        records = data
    start = args.start_index or 0
    end = args.end_index or len(records)
    return records[start:end]


def normalize_record(args: argparse.Namespace, record: dict[str, Any], smoke_index: int) -> dict[str, Any]:
    if not args.smoke:
        return record

    prompt_short = record.get("prompt", "")
    return {
        "prompt": prompt_short,
        "long_prompt": record.get("long_prompt", prompt_short),
        "negative_prompt": record.get("negative_prompt"),
        "image": str(resolve_path(record["gt"])),
        "edit_image": str(resolve_path(record["background"])),
        "generated_subject_image": str(resolve_path(record["subject"])),
        "sub": str(resolve_path(record["subcrop"])),
        "back_mask": str(resolve_path(record["mask"])),
        "item_idx": record.get("item_idx", smoke_index),
        "subject_idx": record.get("subject_idx", smoke_index),
        "prompt_id": record.get("prompt_id", ""),
        "seed": record.get("best_seed"),
        "_smoke_index": smoke_index,
    }


def mask_to_tensor(path: Path, width: int, height: int) -> torch.Tensor:
    """Load a pixel-space mask and align it to the main image's (width, height).

    Uses the same aspect-preserving resize/center-crop geometry as the main image so
    mask tokens stay registered to the main-image latent grid, but with NEAREST
    resampling to keep the mask binary.
    """
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("L")
    image = _resize_and_crop(image, width, height, Image.Resampling.NEAREST)
    tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).view(height, width).to(torch.float32)
    return tensor.contiguous()


@torch.no_grad()
def encode_patchified_latents(
    vae: AutoencoderKLFlux2,
    pixel_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pixel_values = pixel_values.to(device=vae.device, dtype=vae.dtype)
    raw_latents = vae.encode(pixel_values).latent_dist.mode()
    patchified = Flux2KleinPipeline._patchify_latents(raw_latents)
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device=patchified.device, dtype=patchified.dtype)
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        device=patchified.device,
        dtype=patchified.dtype,
    )
    patchified = (patchified - bn_mean) / bn_std
    packed = Flux2KleinPipeline._pack_latents(patchified)
    return patchified.cpu(), packed.cpu()


def build_subcrop_sparse_tensors(
    subcrop_packed: torch.Tensor,
    subcrop_latent_ids: torch.Tensor,
    ref_token_indices: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    if ref_token_indices is None:
        raise ValueError("Expected selection.ref_token_indices for subcrop sparse cache generation.")

    ref_token_indices = ref_token_indices.to(dtype=torch.long, device=subcrop_packed.device).view(-1)
    if subcrop_packed.dim() != 3:
        raise ValueError(f"Expected subcrop_packed_latents shape [B, L, C], got {tuple(subcrop_packed.shape)}.")
    if subcrop_latent_ids.dim() != 3:
        raise ValueError(f"Expected subcrop_latent_ids shape [B, L, D], got {tuple(subcrop_latent_ids.shape)}.")
    if subcrop_packed.shape[:2] != subcrop_latent_ids.shape[:2]:
        raise ValueError(
            "subcrop_packed_latents and subcrop_latent_ids must share batch/sequence dimensions, "
            f"got {tuple(subcrop_packed.shape)} and {tuple(subcrop_latent_ids.shape)}."
        )
    if ref_token_indices.numel() and int(ref_token_indices.max().item()) >= subcrop_packed.shape[1]:
        raise ValueError(
            f"selection.ref_token_indices max {int(ref_token_indices.max().item())} "
            f"exceeds subcrop seq len {subcrop_packed.shape[1]}."
        )

    return {
        "subcrop_sparse_packed_latents": subcrop_packed.index_select(1, ref_token_indices),
        "subcrop_sparse_latent_ids": subcrop_latent_ids.index_select(1, ref_token_indices),
        "subcrop_sparse_token_indices": ref_token_indices.clone(),
    }


def load_models(args: argparse.Namespace):
    device = torch.device(args.device)
    dtype = to_dtype(args.dtype)
    layers = tuple(int(x.strip()) for x in args.text_encoder_out_layers.split(",") if x.strip())

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        local_files_only=args.local_files_only,
    )
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    text_encoder.to(device)
    text_encoder.eval()

    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    vae.to(device)
    vae.eval()

    return tokenizer, text_encoder, vae, device, dtype, layers


@torch.no_grad()
def cache_one_sample(
    args: argparse.Namespace,
    record: dict[str, Any],
    sample_index: int,
    out_dir: Path,
    tokenizer: Qwen2TokenizerFast,
    text_encoder: Qwen3ForCausalLM,
    vae: AutoencoderKLFlux2,
    device: torch.device,
    dtype: torch.dtype,
    layers: tuple[int, ...],
) -> dict[str, Any]:
    prompt_short = record["prompt"]
    prompt_long = record.get("long_prompt", "")
    negative_prompt = record.get("negative_prompt")

    target_path = Path(record["image"])
    background_path = Path(record["edit_image"])
    subject_path = Path(record["generated_subject_image"])
    subcrop_path = Path(record["sub"])
    mask_path = Path(record["back_mask"])

    all_tensors: dict[str, torch.Tensor] = {}
    manifest_meta: dict[str, Any] = {}

    long_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=prompt_long,
        device=device,
        dtype=dtype,
        max_sequence_length=args.max_sequence_length,
        hidden_states_layers=layers,
    )
    long_text_ids = Flux2KleinPipeline._prepare_text_ids(long_embeds).to(device)
    all_tensors["long_prompt_embeds"] = long_embeds.cpu()
    all_tensors["long_text_ids"] = long_text_ids.cpu()

    prompt_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=prompt_short,
        device=device,
        dtype=dtype,
        max_sequence_length=args.max_sequence_length,
        hidden_states_layers=layers,
    )
    text_ids = Flux2KleinPipeline._prepare_text_ids(prompt_embeds).to(device)
    all_tensors["prompt_embeds"] = prompt_embeds.cpu()
    all_tensors["text_ids"] = text_ids.cpu()

    manifest_meta["prompt"] = {
        "prompt": prompt_short,
        "long_prompt": prompt_long,
        "negative_prompt": negative_prompt,
        "text_encoder_out_layers": list(layers),
        "max_sequence_length": args.max_sequence_length,
    }

    multiple_of = 16
    max_area = args.resolution * args.resolution

    # The main scene (target / background / mask) shares one 16-aligned resolution,
    # derived from the target image, so the mask tokens stay registered to the main
    # image latent grid. The subject image is aligned independently. No image is
    # stretched to a fixed square: the original resolution / aspect ratio is kept and
    # only floored to a multiple of 16, matching the Flux2 Klein pipeline.
    patchified_by_role: dict[str, torch.Tensor] = {}

    target_pixels, main_w, main_h = preprocess_aligned_image(
        target_path, multiple_of=multiple_of, max_area=max_area)
    background_pixels, _, _ = preprocess_aligned_image(
        background_path, multiple_of=multiple_of, max_area=max_area, force_size=(main_w, main_h))
    subject_pixels, _, _ = preprocess_aligned_image(
        subject_path, multiple_of=multiple_of, max_area=max_area)

    role_pixels: dict[str, torch.Tensor] = {
        "target": target_pixels,
        "background": background_pixels,
        "subject": subject_pixels,
    }
    for role, pixels in role_pixels.items():
        patchified, packed = encode_patchified_latents(vae, pixels)
        latent_ids = Flux2KleinPipeline._prepare_latent_ids(patchified)
        all_tensors[f"{role}_patchified_latents"] = patchified
        all_tensors[f"{role}_packed_latents"] = packed
        all_tensors[f"{role}_latent_ids"] = latent_ids
        manifest_meta[role] = {
            "role": role,
            "pixels_size": [int(pixels.shape[-2]), int(pixels.shape[-1])],
        }
        patchified_by_role[role] = patchified

    mask_tensor = mask_to_tensor(mask_path, main_w, main_h)
    selection = build_sparse_token_selection_from_mask(mask_tensor, downsample_factor=16, threshold=0.5)
    if selection.crop_bounds is None:
        raise ValueError(f"Sample {sample_index}: mask produced empty crop_bounds")

    y1, y2, x1, x2 = selection.crop_bounds
    crop_h_tokens = y2 - y1
    crop_w_tokens = x2 - x1
    crop_h_pixels = crop_h_tokens * 16
    crop_w_pixels = crop_w_tokens * 16

    subcrop_pixels = image_to_tensor(subcrop_path, width=crop_w_pixels, height=crop_h_pixels)
    subcrop_patchified, subcrop_packed = encode_patchified_latents(vae, subcrop_pixels)
    subcrop_latent_ids = Flux2KleinPipeline._prepare_latent_ids(subcrop_patchified)
    subcrop_latent_ids[..., 0] = 60
    all_tensors["subcrop_patchified_latents"] = subcrop_patchified
    all_tensors["subcrop_packed_latents"] = subcrop_packed
    all_tensors["subcrop_latent_ids"] = subcrop_latent_ids
    all_tensors.update(
        build_subcrop_sparse_tensors(
            subcrop_packed,
            subcrop_latent_ids,
            selection.ref_token_indices,
        )
    )
    manifest_meta["subcrop"] = {
        "role": "subcrop",
        "pixels_size": [crop_h_pixels, crop_w_pixels],
        "crop_bounds": [y1, y2, x1, x2],
        "sparse_token_count": int(selection.ref_token_indices.numel() if selection.ref_token_indices is not None else 0),
    }

    for k in ("main_token_indices", "ref_token_indices", "crop_token_indices",
              "token_mask", "token_coords", "crop_token_coords"):
        val = getattr(selection, k, None)
        if val is not None:
            all_tensors[f"selection_{k}"] = val
    all_tensors["selection_mask_tensor"] = mask_tensor
    manifest_meta["selection"] = {
        "crop_bounds": [y1, y2, x1, x2],
        "crop_pixels_size": [crop_h_pixels, crop_w_pixels],
    }

    cond_image_ids = Flux2KleinPipeline._prepare_image_ids(
        [patchified_by_role["background"], patchified_by_role["subject"]],
        scale=20,
    )
    all_tensors["cond_image_ids"] = cond_image_ids

    out_dir.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file({k: v.contiguous().cpu() for k, v in all_tensors.items()},
                                str(out_dir / "cache.safetensors"))

    manifest_keys = [
        "prompt", "long_prompt", "image", "sub", "edit_image",
        "generated_subject_image", "back_mask", "item_idx", "subject_idx",
        "prompt_id", "negative_prompt", "edit_type", "seed",
        "cleaned_subject_description", "subject_role",
        "best_of_three_best_seed", "best_of_three_group_key",
    ]
    manifest_record = {k: record[k] for k in manifest_keys if k in record}

    manifest = {
        "sample_index": sample_index,
        "max_area_resolution": args.resolution,
        "main_size": [main_h, main_w],
        "model": args.pretrained_model_name_or_path,
        "dtype": args.dtype,
        "device": str(device),
        "cache_safetensors": "cache.safetensors",
        "record": manifest_record,
        **manifest_meta,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    return manifest


def is_cached(out_dir: Path) -> bool:
    return (out_dir / "manifest.json").exists() and (out_dir / "cache.safetensors").exists()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_records = load_records(args)
    total = len(raw_records)
    start_index = args.start_index or 0
    print(f"Loaded {total} records (range [{start_index}, {start_index + total})) from {args.dataset_json}")

    tokenizer, text_encoder, vae, device, dtype, layers = load_models(args)
    print(f"Models loaded on {device}, dtype={args.dtype}")

    chunk_tag = f"_{start_index:06d}_{start_index + total:06d}" if args.start_index is not None else ""
    global_manifest_path = output_dir / f"global_manifest{chunk_tag}.jsonl"
    existing_indices: set[int] = set()

    if args.resume and global_manifest_path.exists():
        with global_manifest_path.open() as f:
            for line in f:
                entry = json.loads(line)
                existing_indices.add(entry["index"])
        print(f"Resume mode: {len(existing_indices)} samples already cached, skipping")

    skipped = 0
    processed = 0
    errors: list[dict] = []

    global_f = global_manifest_path.open("a" if args.resume else "w")

    pbar = tqdm(range(total), desc="Caching", unit="sample")
    for local_idx in pbar:
        global_idx = start_index + local_idx
        if args.resume and global_idx in existing_indices:
            skipped += 1
            pbar.set_postfix_str(f"skipped={skipped}, done={processed}")
            continue

        sample_dir = output_dir / f"{global_idx:06d}"
        if is_cached(sample_dir) and args.resume:
            skipped += 1
            pbar.set_postfix_str(f"skipped={skipped}, done={processed}")
            global_f.write(json.dumps({"index": global_idx, "manifest": f"{global_idx:06d}/manifest.json"}, ensure_ascii=False) + "\n")
            global_f.flush()
            continue

        try:
            raw_record = raw_records[local_idx]
            record = normalize_record(args, raw_record, global_idx)
            manifest = cache_one_sample(
                args, record, global_idx, sample_dir,
                tokenizer, text_encoder, vae, device, dtype, layers,
            )
            line = {
                "index": global_idx,
                "manifest": f"{global_idx:06d}/manifest.json",
                "item_idx": record.get("item_idx"),
                "subject_idx": record.get("subject_idx"),
                "prompt_id": record.get("prompt_id"),
            }
            global_f.write(json.dumps(line, ensure_ascii=False) + "\n")
            global_f.flush()
            processed += 1
            pbar.set_postfix_str(f"skipped={skipped}, done={processed}")
        except Exception as exc:
            errors.append({"index": global_idx, "item_idx": raw_record.get("item_idx"), "error": str(exc)})
            pbar.set_postfix_str(f"err={len(errors)}, done={processed}")
            print(f"\n[ERROR] idx={global_idx}: {exc}")

        if processed > 0 and processed % args.gc_every == 0:
            import gc
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    global_f.close()

    if errors:
        errors_path = output_dir / f"cache_errors{chunk_tag}.json"
        errors_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2))
        print(f"\n{len(errors)} errors written to {errors_path}")

    print(f"\nDone: {processed} cached, {skipped} skipped, {len(errors)} errors")
    print(f"Global manifest: {global_manifest_path}")


if __name__ == "__main__":
    main()
