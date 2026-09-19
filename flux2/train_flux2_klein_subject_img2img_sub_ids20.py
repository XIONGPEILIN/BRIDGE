#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import inspect
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch._dynamo
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from datetime import timedelta
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
from safetensors.torch import load_file as safetensors_load_file


REPO_ROOT = Path(__file__).resolve().parent
LOCAL_DIFFUSERS_SRC = REPO_ROOT / "diffusers" / "src"
if LOCAL_DIFFUSERS_SRC.exists() and str(LOCAL_DIFFUSERS_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIFFUSERS_SRC))

import diffusers  # noqa: E402
from diffusers import (  # noqa: E402
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
)
from diffusers.training_utils import (  # noqa: E402
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import check_min_version  # noqa: E402
from diffusers.utils.torch_utils import is_compiled_module  # noqa: E402

from flux2_klein_pe_exchange import Flux2KleinPEExchangeTransformer2DModel  # noqa: E402
from flux2_klein_pe_soft_exchange import Flux2KleinPESoftExchangeTransformer2DModel  # noqa: E402
from qwen_pe_exchange_sparse_model import (  # noqa: E402
    SparseTokenSelection,
    build_bbox_pe_exchange_selection,
    build_sparse_token_selection_from_mask,
    select_sparse_tokens,
)


check_min_version("0.39.0.dev0")

logger = get_logger(__name__)


DEFAULT_DATASET = REPO_ROOT / "qwen" / "subject_dataset_runs" / "all" / "dataset_qwen_subject_driven_all.json"
DEFAULT_TARGET_BASE = REPO_ROOT / "qwen" / "subject_dataset_runs" / "all"
DEFAULT_SOURCE_BASE = REPO_ROOT / "qwen" / "picobanana" / "openimages"
DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-base-9B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FLUX2 Klein subject-driven img2img with sub branch at t=20 and conditions at t=40+."
    )
    parser.add_argument("--pretrained_model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--target_base", type=Path, default=DEFAULT_TARGET_BASE)
    parser.add_argument("--source_base", type=Path, default=DEFAULT_SOURCE_BASE)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--experiment_mode",
        choices=("no_sub", "sub_no_exchange", "soft_exchange", "hard_exchange"),
        default="no_sub",
    )
    parser.add_argument("--sub_region_mode", choices=("bbox", "mask"), default="mask")
    parser.add_argument(
        "--pe_exchange_region",
        choices=("mask", "bbox"),
        default="mask",
        help="Cache mode PE mapping. `mask` preserves the existing mask-only main/ref pairs. "
             "`bbox` pairs every main-image bbox token with the row-major dense sub token; "
             "requires the dense sub branch (no --use_sparse_sub_branch).",
    )
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--deepspeed_compile",
        action="store_true",
        help="DeepSpeed path only. Explicitly call the DeepSpeed engine's torch.compile "
             "(max-autotune-no-cudagraphs) after prepare(). Needed because accelerate's auto "
             "dynamo path does not actually fire max-autotune under DeepSpeed. No-op without "
             "a DeepSpeed plugin (the plain-DDP path uses accelerate's own dynamo config).",
    )
    parser.add_argument(
        "--grad_accum_dtype",
        type=str,
        default=None,
        choices=("fp32", "bf16", "fp16"),
        help="DeepSpeed path only. Sets data_types.grad_accum_dtype. DeepSpeed ZeRO defaults to "
             "fp32 gradient accumulation (an extra fp32 grad buffer ~= params*4B/world ≈ 9GB/GPU "
             "on 9B/4-GPU), which can OOM at the memory ceiling. 'bf16' drops that buffer (no fp32 "
             "accumulation copy) at the cost of slightly noisier gradients across the accumulation "
             "window. No-op without a DeepSpeed plugin.",
    )
    parser.add_argument(
        "--gc_use_reentrant",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use reentrant gradient checkpointing (torch.utils.checkpoint use_reentrant=True) "
             "instead of diffusers' default use_reentrant=False. Needed to make torch.compile "
             "(max-autotune) work with GC: the non-reentrant default trips the dynamo "
             "`lift_tracked_freevar_to_input should not be called on root SubgraphTracer` "
             "assertion when tracing the checkpoint closure.",
    )
    parser.add_argument("--learning_rate", type=float, default=1.0)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--optimizer", choices=("prodigyplus_schedulefree", "adamw"), default="prodigyplus_schedulefree")
    parser.add_argument("--d0", type=float, default=1e-6)
    parser.add_argument("--prodigy_steps", type=int, default=10000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--text_encoder_out_layers", default="9,18,27")
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument(
        "--weighting_scheme",
        choices=("sigma_sqrt", "logit_normal", "mode", "cosmap", "none"),
        default="none",
    )
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--sub_loss_weight", type=float, default=1.0)
    parser.add_argument("--checkpointing_steps", type=int, default=200)
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--logging_dir", default="logs")
    parser.add_argument("--report_to", choices=("none", "tensorboard", "wandb"), default="tensorboard")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--subject_image_field", choices=("ref_gt_crop", "edit_image"), default="ref_gt_crop")
    parser.add_argument("--ste_encoder_layers", type=int, default=1)
    parser.add_argument("--ste_encoder_num_heads", type=int, default=8)
    parser.add_argument("--ste_head_init", choices=("zero", "normal"), default="zero")
    parser.add_argument(
        "--bucket_order_file",
        type=Path,
        default=None,
        help="Optional JSON produced by precompute_bucket_order.py. When set, training reads "
             "batches in this fixed order (same order every epoch) instead of the dynamic "
             "BucketBatchSampler. Required for deterministic resume with train_batch_size > 1.",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Path to cache_full/. When set, training reads pre-cached VAE latents, prompt "
             "embeds, and sparse selection from per-sample safetensors instead of raw images "
             "(skipping VAE encode, text encoder forward, and mask reads). Enables padding "
             "+ attention_mask path for variable sub_seq_len batches.",
    )
    parser.add_argument(
        "--use_long_prompt",
        action="store_true",
        help="Cache mode only. Use cached `long_prompt_embeds` / `long_text_ids` instead of "
             "the short `prompt_embeds` / `text_ids`.",
    )
    parser.add_argument(
        "--long_prompt_prob",
        type=float,
        default=None,
        help="Cache mode only. Probability in [0, 1] of using the cached LONG prompt for each "
             "sample; otherwise the SHORT prompt is used. When set, this OVERRIDES "
             "--use_long_prompt and picks long-vs-short RANDOMLY PER SAMPLE (1.0 = always long, "
             "0.0 = always short, 0.5 = free mix). Long and short prompts have different "
             "sequence lengths and cannot be stacked, so random mixing requires "
             "train_batch_size == 1.",
    )
    parser.add_argument(
        "--subject_drop_prob",
        type=float,
        default=0.0,
        help="Probability in [0, 1] of dropping the subject condition for each step "
             "(classifier-free-guidance-style dropout). When dropped, the input is only "
             "prompt + background image: no subject reference latent, no sub branch, and no PE "
             "exchange (the step behaves like experiment_mode=no_sub). Decided per batch "
             "(== per sample at the bs=1 sparse config). 0.0 disables (default).",
    )
    parser.add_argument(
        "--prompt_drop_prob",
        type=float,
        default=0.0,
        help="Probability in [0, 1] of dropping the TEXT prompt PER SAMPLE for classifier-free "
             "guidance (CFG) training. Klein-base inference does two-pass text CFG with an empty "
             "negative prompt, so training replaces the prompt embeds with the encoded empty-string "
             "('') NULL embedding (see precompute_null_prompt.py) for the dropped samples — image "
             "conditions (subject / background / sub branch / PE exchange) are KEPT. 0.0 disables "
             "(default); typical CFG value 0.1. Cache mode only (uses the cached/null embeds).",
    )
    parser.add_argument(
        "--null_prompt_embeds_path",
        type=str,
        default=None,
        help="Path to the empty-prompt NULL embedding (precompute_null_prompt.py output). Required "
             "when --prompt_drop_prob > 0. Defaults to <cache_dir>/_null_prompt_embeds.safetensors.",
    )
    parser.add_argument(
        "--use_sparse_sub_branch",
        action="store_true",
        help="Cache mode only. Sub branch input becomes the sparse set of subject-crop tokens "
             "selected by `selection.ref_token_indices` (length = number of mask-hit tokens) "
             "instead of the dense crop rectangle. PE exchange becomes a 1-to-1 swap "
             "(ref_token_indices=arange(N_sparse)). Cached collation pads variable lengths, "
             "so train_batch_size > 1 is supported.",
    )
    parser.add_argument(
        "--zero_cond_t",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When set, condition tokens (background + subject conditioning latents at the end "
             "of the image stream) receive AdaLN modulation derived from timestep=0 instead of "
             "the denoising timestep. Matches Qwen image 2511's `zero_cond_t` design.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def parse_text_layers(spec: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in spec.split(",") if x.strip())


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized


def image_to_tensor(path: Path, resolution: int) -> torch.Tensor:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((resolution, resolution), Image.Resampling.BILINEAR)
    tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).view(resolution, resolution, 3)
    tensor = tensor.permute(2, 0, 1).to(torch.float32).div_(255.0)
    return tensor.mul_(2.0).sub_(1.0).contiguous()


def mask_to_tensor(path: Path, resolution: int) -> torch.Tensor:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("L")
    image = image.resize((resolution, resolution), Image.Resampling.NEAREST)
    tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).view(resolution, resolution)
    return tensor.to(torch.float32).div_(255.0).contiguous()


class SubjectDrivenFlux2Dataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        target_base: Path,
        source_base: Path,
        resolution: int,
        subject_image_field: str = "ref_gt_crop",
        max_train_samples: Optional[int] = None,
        downsample_factor: int = 16,
        precompute_bucket_keys: bool = True,
        sub_region_mode: str = "mask",
    ):
        self.dataset_path = resolve_path(dataset_path)
        self.target_base = resolve_path(target_base)
        self.source_base = resolve_path(source_base)
        self.resolution = resolution
        self.subject_image_field = subject_image_field
        self.downsample_factor = downsample_factor
        self.sub_region_mode = sub_region_mode

        with self.dataset_path.open() as f:
            rows = json.load(f)
        if max_train_samples is not None:
            rows = rows[:max_train_samples]

        if precompute_bucket_keys:
            if max_train_samples is None:
                # Full-dataset run: read-through cache (one-time mask scan, ~5 min for 30k samples).
                self.bucket_keys = self._load_or_compute_bucket_keys(rows)
            else:
                # Sliced run (smoke / debug): cheap inline compute, skip cache.
                self.bucket_keys = self._compute_bucket_keys_inline(rows)
        else:
            self.bucket_keys = [(0, 0) for _ in rows]
        self.rows = rows

    def _bucket_cache_path(self) -> Path:
        return self.dataset_path.with_name(
            f"{self.dataset_path.stem}_bucket_keys_res{self.resolution}_ds{self.downsample_factor}.json"
        )

    def _compute_bucket_keys_inline(
        self, rows: list[dict[str, Any]]
    ) -> list[tuple[int, int]]:
        keys: list[tuple[int, int]] = []
        iterator = tqdm(range(len(rows)), desc="Bucket keys", disable=len(rows) < 32)
        for idx in iterator:
            row = rows[idx]
            mask_path = self.source_base / row["back_mask"]
            try:
                mask = mask_to_tensor(mask_path, self.resolution)
                selection = build_sparse_token_selection_from_mask(
                    mask, downsample_factor=self.downsample_factor, threshold=0.5
                )
                if selection.crop_bounds is None:
                    keys.append((0, 0))
                else:
                    y1, y2, x1, x2 = selection.crop_bounds
                    keys.append((int(y2 - y1), int(x2 - x1)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Bucket key compute failed for sample %d (%s): %s", idx, mask_path, exc)
                keys.append((0, 0))
        return keys

    def _load_or_compute_bucket_keys(
        self, rows: list[dict[str, Any]]
    ) -> list[tuple[int, int]]:
        cache_path = self._bucket_cache_path()
        if cache_path.exists():
            try:
                with cache_path.open() as f:
                    cached = json.load(f)
                if isinstance(cached, list) and len(cached) == len(rows):
                    return [tuple(item) for item in cached]
                logger.warning(
                    "Bucket key cache length %d does not match dataset length %d, recomputing.",
                    len(cached) if isinstance(cached, list) else -1,
                    len(rows),
                )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to read bucket key cache %s: %s", cache_path, exc)

        logger.info(
            "Precomputing bucket keys for %d samples (one-time cost). This reads every mask once.",
            len(rows),
        )
        keys = self._compute_bucket_keys_inline(rows)
        try:
            with cache_path.open("w") as f:
                json.dump(keys, f)
            logger.info("Saved bucket key cache to %s", cache_path)
        except OSError as exc:
            logger.warning("Failed to write bucket key cache %s: %s", cache_path, exc)
        return keys

    def __len__(self) -> int:
        return len(self.rows)

    def _subject_path(self, row: dict[str, Any]) -> Path:
        if self.subject_image_field == "ref_gt_crop" and row.get("ref_gt_crop"):
            return self.source_base / row["ref_gt_crop"]
        edit_image = row.get("edit_image") or []
        if not edit_image:
            raise ValueError("Row has no edit_image and no ref_gt_crop.")
        return self.source_base / edit_image[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        target_path = self.target_base / row["image"]
        main_path = self.source_base / row["ref_gt"]
        subject_path = self._subject_path(row)
        mask_path = self.source_base / row["back_mask"]

        mask_tensor = mask_to_tensor(mask_path, self.resolution)
        if self.sub_region_mode == "bbox":
            selection = build_bbox_token_selection_from_mask(
                mask_tensor, downsample_factor=self.downsample_factor, threshold=0.5
            )
        else:
            selection = build_sparse_token_selection_from_mask(
                mask_tensor, downsample_factor=self.downsample_factor, threshold=0.5
            )

        return {
            "prompt": row["prompt"],
            "target_pixels": image_to_tensor(target_path, self.resolution),
            "main_pixels": image_to_tensor(main_path, self.resolution),
            "subject_pixels": image_to_tensor(subject_path, self.resolution),
            "mask": mask_tensor,
            "selection": selection,
            "meta": {
                "index": index,
                "item_idx": row.get("item_idx"),
                "subject_idx": row.get("subject_idx"),
                "seed": row.get("seed"),
                "target_path": str(target_path),
                "main_path": str(main_path),
                "subject_path": str(subject_path),
                "mask_path": str(mask_path),
            },
        }


def collate_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prompts": [example["prompt"] for example in examples],
        "target_pixels": torch.stack([example["target_pixels"] for example in examples], dim=0),
        "main_pixels": torch.stack([example["main_pixels"] for example in examples], dim=0),
        "subject_pixels": torch.stack([example["subject_pixels"] for example in examples], dim=0),
        "masks": torch.stack([example["mask"] for example in examples], dim=0),
        "selections": [example["selection"] for example in examples],
        "meta": [example["meta"] for example in examples],
    }


class BucketBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Yield batches whose samples share the same (crop_h_tokens, crop_w_tokens) bucket.

    Why: the dense sub-branch token count equals `crop_h * crop_w`, which varies per
    sample. Bucketing by that pair guarantees all samples in a batch have the same
    `sub_seq_len` so they can be stacked along the batch dim. The PE swap still
    differs per sample because the absolute (y1, x1) of each crop differs.

    Shuffling uses the global `random` module (already seeded via
    `accelerate.utils.set_seed` at startup), so the batch order is deterministic
    given the training seed but evolves naturally across epochs.
    """

    def __init__(
        self,
        dataset: "SubjectDrivenFlux2Dataset",
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
    ):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive int, got {batch_size}.")
        if not hasattr(dataset, "bucket_keys"):
            raise AttributeError("Dataset must expose `bucket_keys` for BucketBatchSampler.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        bucket_to_indices: dict[tuple[int, int], list[int]] = {}
        skipped_empty = 0
        for idx, key in enumerate(dataset.bucket_keys):
            key = tuple(key)
            if key == (0, 0):
                # Empty mask -> sub-branch would be empty; skip these samples.
                skipped_empty += 1
                continue
            bucket_to_indices.setdefault(key, []).append(idx)
        self._bucket_to_indices = bucket_to_indices

        if skipped_empty > 0:
            logger.info(
                "BucketBatchSampler: skipped %d samples with empty mask bucket (0, 0).",
                skipped_empty,
            )

        self._build_batches()

    def _build_batches(self) -> None:
        batches: list[list[int]] = []
        for indices in self._bucket_to_indices.values():
            order = list(indices)
            if self.shuffle:
                random.shuffle(order)
            for i in range(0, len(order), self.batch_size):
                batch = order[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)
        self._batches = batches

    def __iter__(self):
        if self.shuffle:
            self._build_batches()
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


class PrecomputedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Yield a fixed list of pre-built batches loaded from a JSON sidecar.

    Used in conjunction with `precompute_bucket_order.py` to make the per-epoch
    batch ordering deterministic across runs and resumes. The same `batches`
    list is replayed every epoch.
    """

    def __init__(self, batches: list[list[int]]):
        self._batches = [list(b) for b in batches]

    def __iter__(self):
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


def load_precomputed_batches(
    path: Path,
    dataset_size: int,
    expected_batch_size: int,
) -> list[list[int]]:
    """Load and validate a JSON file written by precompute_bucket_order.py.

    With sort+greedy packing the LAST batch may be smaller than
    `expected_batch_size`; that's allowed (collate will still pad sub to batch max).
    """
    with path.open() as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "batches" not in payload:
        raise ValueError(f"Bucket order file {path} is missing the 'batches' field.")
    batches = payload["batches"]
    if not isinstance(batches, list) or not all(isinstance(b, list) for b in batches):
        raise ValueError(f"Bucket order file {path} has malformed 'batches' field.")

    meta = payload.get("meta", {})
    file_bs = meta.get("train_batch_size")
    if file_bs is not None and file_bs != expected_batch_size:
        raise ValueError(
            f"Bucket order file {path} was built with train_batch_size={file_bs}, "
            f"but training was launched with train_batch_size={expected_batch_size}. "
            f"Re-run precompute_bucket_order.py with --train_batch_size {expected_batch_size}."
        )

    # Sanity-check batch sizes: all but possibly the last should equal expected_batch_size.
    oversize = [i for i, b in enumerate(batches) if len(b) > expected_batch_size]
    if oversize:
        raise ValueError(
            f"Bucket order file {path}: batches at indices {oversize[:3]}... have size > "
            f"{expected_batch_size}; the file is corrupt or was built for a different batch size."
        )

    max_idx = max((max(b) for b in batches if b), default=-1)
    if max_idx >= dataset_size:
        raise ValueError(
            f"Bucket order file {path} references index {max_idx} but the dataset only has "
            f"{dataset_size} samples. The file is stale; re-run precompute_bucket_order.py."
        )
    return batches


CACHE_LOOKUP_FILENAME = "_lookup_by_item_subject.json"
CACHE_BUCKET_KEYS_FILENAME = "_bucket_keys_from_cache.json"
CACHE_MAIN_KEYS_FILENAME = "_main_keys_from_cache.json"


def build_cache_lookup(cache_dir: Path) -> dict[str, int]:
    """Map `"<item_idx>_<subject_idx>"` -> shard index by scanning every cache manifest.

    Cached to `<cache_dir>/_lookup_by_item_subject.json` on first call (~30 s) and reloaded
    on subsequent calls (instant).
    """
    cache_dir = resolve_path(cache_dir)
    lookup_path = cache_dir / CACHE_LOOKUP_FILENAME
    if lookup_path.exists():
        with lookup_path.open() as f:
            return json.load(f)

    logger.info("Building cache lookup by scanning manifests in %s (one-time)...", cache_dir)
    shard_dirs = sorted(p for p in cache_dir.iterdir() if p.is_dir() and p.name.isdigit())
    lookup: dict[str, int] = {}
    for shard in tqdm(shard_dirs, desc="Cache lookup"):
        with (shard / "manifest.json").open() as f:
            m = json.load(f)
        rec = m["record"]
        lookup[f"{rec['item_idx']}_{rec['subject_idx']}"] = int(shard.name)
    with lookup_path.open("w") as f:
        json.dump(lookup, f)
    logger.info("Saved cache lookup (%d entries) to %s", len(lookup), lookup_path)
    return lookup


def load_cache_bucket_keys(cache_dir: Path) -> list[tuple[int, int]]:
    """Read `(crop_h_tokens, crop_w_tokens)` for each shard from cached manifests.

    Result is in shard order (shard 0, 1, 2, ...), NOT dataset/split order. Caller must
    re-map via `build_cache_lookup` if needed. Cached to disk after first call.
    """
    cache_dir = resolve_path(cache_dir)
    cache_path = cache_dir / CACHE_BUCKET_KEYS_FILENAME
    if cache_path.exists():
        with cache_path.open() as f:
            data = json.load(f)
        return [tuple(k) for k in data]

    logger.info("Computing bucket keys from cache manifests in %s (one-time)...", cache_dir)
    shard_dirs = sorted(p for p in cache_dir.iterdir() if p.is_dir() and p.name.isdigit())
    keys: list[tuple[int, int]] = []
    for shard in tqdm(shard_dirs, desc="Bucket keys"):
        with (shard / "manifest.json").open() as f:
            m = json.load(f)
        h_px, w_px = m["subcrop"]["pixels_size"]
        keys.append((h_px // 16, w_px // 16))
    with cache_path.open("w") as f:
        json.dump(keys, f)
    return keys


def load_cache_main_keys(cache_dir: Path) -> list[tuple[int, int]]:
    """Read `(target_h_tokens, target_w_tokens)` for each shard from cached manifests.

    This is the MAIN-image (target/background) token grid. It is the bucketing key for
    train_batch_size > 1: `collate_cached_examples` hard-`torch.stack`s `packed_target`,
    `packed_main` and `cond_image_ids`, so every sample in a batch must share the same main
    token count. The cache is native multi-aspect (10 fixed area-capped resolutions), and
    background pixels == target pixels for every sample, so grouping by the target grid also
    makes `cond_image_ids` (= background_tokens + 4096 subject) homogeneous within a batch.

    Result is in shard order, NOT dataset/split order. Caller must re-map via
    `build_cache_lookup` if needed. Cached to disk after first call.
    """
    cache_dir = resolve_path(cache_dir)
    cache_path = cache_dir / CACHE_MAIN_KEYS_FILENAME
    if cache_path.exists():
        with cache_path.open() as f:
            data = json.load(f)
        return [tuple(k) for k in data]

    logger.info("Computing MAIN bucket keys from cache manifests in %s (one-time)...", cache_dir)
    shard_dirs = sorted(p for p in cache_dir.iterdir() if p.is_dir() and p.name.isdigit())
    keys: list[tuple[int, int]] = []
    for shard in tqdm(shard_dirs, desc="Main keys"):
        with (shard / "manifest.json").open() as f:
            m = json.load(f)
        h_px, w_px = m["target"]["pixels_size"]
        keys.append((h_px // 16, w_px // 16))
    with cache_path.open("w") as f:
        json.dump(keys, f)
    return keys


class CachedSubjectDrivenFlux2Dataset(Dataset):
    """Dataset backed by `cache_full/{shard:06d}/cache.safetensors`.

    `__getitem__` loads one shard's full safetensors blob, reconstructs the
    `SparseTokenSelection` from `selection_*` fields, and applies the t-coord remap
    so the cache (built for the t=60 variant) matches the `_ids20` (t=20) layout:
        subcrop_latent_ids[..., 0] = 20
        cond_image_ids[:, :4096, 0] = 40   # background
        cond_image_ids[:, 4096:, 0] = 60   # subject

    The returned dict has pre-encoded latents and prompt embeds, so the training loop
    can skip the VAE and text encoder entirely.
    """

    def __init__(
        self,
        dataset_path: Path,
        cache_dir: Path,
        max_train_samples: Optional[int] = None,
        precompute_bucket_keys: bool = True,
        use_long_prompt: bool = False,
        long_prompt_prob: Optional[float] = None,
        use_sparse_sub_branch: bool = False,
        pe_exchange_region: str = "mask",
    ):
        self.dataset_path = resolve_path(dataset_path)
        self.cache_dir = resolve_path(cache_dir)
        self.use_long_prompt = use_long_prompt
        # When set, pick long-vs-short prompt randomly per sample (overrides use_long_prompt).
        self.long_prompt_prob = long_prompt_prob
        self.use_sparse_sub_branch = use_sparse_sub_branch
        if pe_exchange_region not in {"mask", "bbox"}:
            raise ValueError(f"Unsupported pe_exchange_region: {pe_exchange_region}.")
        if use_sparse_sub_branch and pe_exchange_region == "bbox":
            raise ValueError(
                "pe_exchange_region='bbox' requires the dense sub branch; "
                "disable use_sparse_sub_branch."
            )
        self.pe_exchange_region = pe_exchange_region

        with self.dataset_path.open() as f:
            rows = json.load(f)
        if max_train_samples is not None:
            rows = rows[:max_train_samples]

        lookup = build_cache_lookup(self.cache_dir)
        shard_indices: list[int] = []
        missing = 0
        for row in rows:
            key = f"{row['item_idx']}_{row['subject_idx']}"
            shard = lookup.get(key)
            if shard is None:
                missing += 1
                shard_indices.append(-1)
            else:
                shard_indices.append(shard)
        if missing > 0:
            logger.warning(
                "Cache lookup missed %d/%d rows in %s; those samples will raise on __getitem__.",
                missing, len(rows), self.dataset_path.name,
            )
        self.rows = rows
        self.shard_indices = shard_indices

        if precompute_bucket_keys:
            all_keys = load_cache_bucket_keys(self.cache_dir)
            self.bucket_keys = [
                all_keys[s] if 0 <= s < len(all_keys) else (0, 0) for s in shard_indices
            ]
            # MAIN-image token grid per sample (target/background). Required to bucket batches
            # by main resolution at train_batch_size > 1 (see load_cache_main_keys). (0, 0) for
            # rows with no cache shard, same convention as bucket_keys.
            all_main_keys = load_cache_main_keys(self.cache_dir)
            self.main_keys = [
                all_main_keys[s] if 0 <= s < len(all_main_keys) else (0, 0) for s in shard_indices
            ]
        else:
            self.bucket_keys = [(0, 0) for _ in rows]
            self.main_keys = [(0, 0) for _ in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard = self.shard_indices[index]
        if shard < 0:
            raise KeyError(f"Row {index} has no cache shard (item_idx={self.rows[index].get('item_idx')}).")
        shard_path = self.cache_dir / f"{shard:06d}" / "cache.safetensors"
        tensors = safetensors_load_file(str(shard_path))

        cond_image_ids = tensors["cond_image_ids"].clone()
        # cache layout puts background at t=20, subject at t=40; _ids20 expects 40/60.
        # The boundary is the ACTUAL background token count (== target tokens, native aspect),
        # NOT a hardcoded 4096: non-1024² samples have <4096 background tokens (e.g. 832x1248 ->
        # 4056), so a fixed 4096 split would mislabel the first (4096 - bg_len) *subject* cond
        # tokens as t=40 instead of t=60. cond_image_ids is [background; subject] with subject
        # always 1024² (4096), so bg_len = background_packed_latents length.
        bg_len = int(tensors["background_packed_latents"].shape[1])
        cond_image_ids[:, :bg_len, 0] = 40   # background
        cond_image_ids[:, bg_len:, 0] = 60   # subject

        subcrop_latent_ids = tensors["subcrop_latent_ids"].clone()
        # cache hard-codes t=60 for sub; _ids20 expects t=20.
        subcrop_latent_ids[..., 0] = 20

        target_latent_ids = tensors["target_latent_ids"].clone()
        target_latent_ids[..., 0] = 0  # _ids20 target is at t=0 (cache already 0, defensive)

        main_token_indices = tensors["selection_main_token_indices"].to(torch.long)
        ref_token_indices = (
            tensors["selection_ref_token_indices"].to(torch.long)
            if "selection_ref_token_indices" in tensors else None
        )
        crop_token_indices = (
            tensors["selection_crop_token_indices"].to(torch.long)
            if "selection_crop_token_indices" in tensors else None
        )

        # Sub-branch input. SPARSE mode (--use_sparse_sub_branch; used by hard_exchange and
        # soft_exchange): the sub branch is the SUBJECT CROP sparsified to the white-mask tokens,
        # read from cache keys `subcrop_sparse_*` (new cache / backfill_subcrop_sparse_tokens.py =
        # subcrop_packed[selection_ref_token_indices]). REPLACED the old
        # `background_packed_latents[main_token_indices]` slice, which wrongly sourced the
        # BACKGROUND image at the subject region (see CLAUDE.md). No fallback: missing keys = stale
        # cache -> hard error.
        if self.use_sparse_sub_branch:
            for _k in ("subcrop_sparse_packed_latents", "subcrop_sparse_latent_ids"):
                if _k not in tensors:
                    raise KeyError(
                        f"shard {shard:06d} is missing '{_k}'. Regenerate/backfill the cache "
                        f"(backfill_subcrop_sparse_tokens.py) before sparse training; "
                        f"there is intentionally NO fallback to the old background-slice path."
                    )
            sub_packed = tensors["subcrop_sparse_packed_latents"].squeeze(0)        # [N_white, D]
            sub_ids_out = tensors["subcrop_sparse_latent_ids"].squeeze(0).clone()   # [N_white, 4]
            sub_ids_out[..., 0] = 20  # cache bakes t=60; the _ids20 sub block lives at t=20
            # ref indices MUST be arange(N_white) for the 1-to-1 PE swap. Do NOT use the cached
            # `subcrop_sparse_token_indices`: it indexes the DENSE subcrop (max ~337 into 340) and
            # would trip the swap guard in qwen_pe_exchange_sparse_model.py (_build_sparse_swapped_freqs).
            ref_token_indices = torch.arange(sub_packed.shape[0], dtype=torch.long)
        else:
            sub_packed = tensors["subcrop_packed_latents"].squeeze(0)
            sub_ids_out = subcrop_latent_ids.squeeze(0)

        selection = SparseTokenSelection(
            main_token_indices=main_token_indices,
            ref_token_indices=ref_token_indices,
            token_mask=tensors.get("selection_token_mask"),
            token_coords=tensors.get("selection_token_coords"),
            crop_token_indices=crop_token_indices,
            crop_token_coords=tensors.get("selection_crop_token_coords"),
            crop_bounds=None,  # not used at training time; recomputable from coords if needed
        )
        if self.pe_exchange_region == "bbox":
            selection = build_bbox_pe_exchange_selection(selection, sub_packed.shape[0])
            crop_coords = selection.crop_token_coords.to(dtype=torch.long)
            main_coords = target_latent_ids.squeeze(0).index_select(
                0, selection.main_token_indices
            )[:, 1:3].to(dtype=torch.long)
            if not torch.equal(main_coords, crop_coords):
                raise ValueError(
                    f"shard {shard:06d} main bbox indices are not aligned with "
                    "selection.crop_token_coords."
                )
            local_sub_coords = sub_ids_out[:, 1:3].to(dtype=torch.long)
            expected_local_coords = crop_coords - crop_coords.amin(dim=0, keepdim=True)
            if not torch.equal(local_sub_coords, expected_local_coords):
                raise ValueError(
                    f"shard {shard:06d} dense sub latent IDs are not row-major aligned "
                    "with the cached main-image bbox coordinates."
                )

        # Prompt embeds: short vs. long. Cache always writes `long_*` alongside.
        # `long_prompt_prob` (when set) overrides the fixed `use_long_prompt` flag and picks
        # randomly per sample, so a single run freely mixes long and short prompts.
        if self.long_prompt_prob is not None:
            use_long = random.random() < self.long_prompt_prob
        else:
            use_long = self.use_long_prompt
        if use_long:
            prompt_embeds_key, text_ids_key = "long_prompt_embeds", "long_text_ids"
            if prompt_embeds_key not in tensors:
                raise KeyError(
                    f"long prompt requested but shard {shard:06d} has no `long_prompt_embeds`. "
                    f"Re-run cache_flux2_full_dataset.py to populate long-prompt fields."
                )
        else:
            prompt_embeds_key, text_ids_key = "prompt_embeds", "text_ids"

        return {
            "is_cached": True,
            "prompt": self.rows[index]["prompt"],
            "packed_target": tensors["target_packed_latents"].squeeze(0),
            "packed_main": tensors["background_packed_latents"].squeeze(0),
            "packed_subject": tensors["subject_packed_latents"].squeeze(0),
            "subcrop_packed": sub_packed,
            "subcrop_latent_ids": sub_ids_out,
            "target_latent_ids": target_latent_ids.squeeze(0),
            "cond_image_ids": cond_image_ids.squeeze(0),
            "prompt_embeds": tensors[prompt_embeds_key].squeeze(0),
            "text_ids": tensors[text_ids_key].squeeze(0),
            "used_long_prompt": bool(use_long),
            "selection": selection,
            "meta": {
                "index": index,
                "shard": shard,
                "pe_exchange_region": self.pe_exchange_region,
                "item_idx": self.rows[index].get("item_idx"),
                "subject_idx": self.rows[index].get("subject_idx"),
            },
        }


def collate_cached_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate cached examples; pad sub-branch tensors to batch max along seq dim.

    Padding is at the end of each sample's sub sequence. Padding positions are flagged
    in `sub_valid_mask [B, max_N_sub]` (bool, True at real tokens).
    """
    bsz = len(examples)
    max_n_sub = max(e["subcrop_packed"].shape[0] for e in examples)
    sub_dim = examples[0]["subcrop_packed"].shape[1]
    id_dim = examples[0]["subcrop_latent_ids"].shape[1]

    sub_packed = torch.zeros((bsz, max_n_sub, sub_dim), dtype=examples[0]["subcrop_packed"].dtype)
    sub_ids = torch.zeros((bsz, max_n_sub, id_dim), dtype=examples[0]["subcrop_latent_ids"].dtype)
    sub_valid_mask = torch.zeros((bsz, max_n_sub), dtype=torch.bool)
    sub_has_padding = False
    for i, ex in enumerate(examples):
        n = ex["subcrop_packed"].shape[0]
        sub_packed[i, :n] = ex["subcrop_packed"]
        sub_ids[i, :n] = ex["subcrop_latent_ids"]
        sub_valid_mask[i, :n] = True
        sub_has_padding = sub_has_padding or n < max_n_sub
        # padded sub_ids rows keep t=20 too so position embed is consistent (will be masked anyway)
        sub_ids[i, n:, 0] = 20

    return {
        "is_cached": True,
        "prompts": [e["prompt"] for e in examples],
        "packed_target": torch.stack([e["packed_target"] for e in examples], dim=0),
        "packed_main": torch.stack([e["packed_main"] for e in examples], dim=0),
        "packed_subject": torch.stack([e["packed_subject"] for e in examples], dim=0),
        "subcrop_packed": sub_packed,
        "subcrop_latent_ids": sub_ids,
        "sub_valid_mask": sub_valid_mask,
        "sub_has_padding": sub_has_padding,
        "target_latent_ids": torch.stack([e["target_latent_ids"] for e in examples], dim=0),
        "cond_image_ids": torch.stack([e["cond_image_ids"] for e in examples], dim=0),
        "prompt_embeds": torch.stack([e["prompt_embeds"] for e in examples], dim=0),
        "text_ids": torch.stack([e["text_ids"] for e in examples], dim=0),
        "used_long_prompts": [bool(e.get("used_long_prompt", False)) for e in examples],
        "selections": [e["selection"] for e in examples],
        "meta": [e["meta"] for e in examples],
    }


def normalize_mask_for_pool(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() == 4:
        pass
    elif mask.dim() not in (2, 3, 4):
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    mask = mask.to(torch.float32)
    if float(mask.max().item()) > 1.0:
        mask = (mask > 128.0).to(torch.float32)
    else:
        mask = (mask > 0.5).to(torch.float32)
    return mask


def build_bbox_token_selection_from_mask(
    mask: torch.Tensor,
    downsample_factor: int = 16,
    threshold: float = 0.5,
) -> SparseTokenSelection:
    mask = normalize_mask_for_pool(mask).to(torch.float32)
    pooled = F.max_pool2d(mask, kernel_size=downsample_factor, stride=downsample_factor, ceil_mode=True)
    token_mask = pooled[0, 0] > threshold
    token_coords = token_mask.nonzero(as_tuple=False)
    device = token_mask.device

    if token_coords.numel() == 0:
        empty = torch.zeros((0,), dtype=torch.long, device=device)
        return SparseTokenSelection(
            main_token_indices=empty,
            ref_token_indices=empty,
            token_mask=token_mask,
            token_coords=token_coords,
            crop_token_indices=empty,
            crop_token_coords=token_coords,
            crop_bounds=None,
        )

    y1 = int(token_coords[:, 0].min().item())
    y2 = int(token_coords[:, 0].max().item()) + 1
    x1 = int(token_coords[:, 1].min().item())
    x2 = int(token_coords[:, 1].max().item()) + 1
    ys = torch.arange(y1, y2, device=device, dtype=torch.long)
    xs = torch.arange(x1, x2, device=device, dtype=torch.long)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    rect_coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=1)
    main_token_indices = rect_coords[:, 0] * token_mask.shape[1] + rect_coords[:, 1]
    ref_token_indices = torch.arange(main_token_indices.shape[0], device=device, dtype=torch.long)

    rect_mask = torch.zeros_like(token_mask, dtype=torch.bool)
    rect_mask[y1:y2, x1:x2] = True
    return SparseTokenSelection(
        main_token_indices=main_token_indices,
        ref_token_indices=ref_token_indices,
        token_mask=rect_mask,
        token_coords=rect_coords,
        crop_token_indices=main_token_indices,
        crop_token_coords=rect_coords,
        crop_bounds=(y1, y2, x1, x2),
    )


def encode_latents(
    vae: AutoencoderKLFlux2,
    pixel_values: torch.Tensor,
    latents_bn_mean: torch.Tensor,
    latents_bn_std: torch.Tensor,
) -> torch.Tensor:
    raw_latents = vae.encode(pixel_values).latent_dist.mode()
    patchified = Flux2KleinPipeline._patchify_latents(raw_latents)
    return (patchified - latents_bn_mean) / latents_bn_std


def build_sparse_sub_branch(
    main_latents: torch.Tensor,
    mask: torch.Tensor,
    sub_region_mode: str,
) -> tuple[SparseTokenSelection, torch.Tensor, torch.Tensor]:
    latent_ids = Flux2KleinPipeline._prepare_latent_ids(main_latents).to(device=main_latents.device)
    packed_main = Flux2KleinPipeline._pack_latents(main_latents)

    if sub_region_mode == "mask":
        selection = build_sparse_token_selection_from_mask(mask, downsample_factor=16, threshold=0.5)
    elif sub_region_mode == "bbox":
        selection = build_bbox_token_selection_from_mask(mask, downsample_factor=16, threshold=0.5)
    else:
        raise ValueError(f"Unsupported sub_region_mode: {sub_region_mode}")

    if selection.main_token_indices.numel() == 0:
        raise ValueError("The selected mask/bbox region produced zero sub tokens.")
    if selection.crop_token_indices is None or selection.crop_token_indices.numel() == 0:
        raise ValueError("The selected mask/bbox region produced zero crop tokens.")

    sub_x0 = select_sparse_tokens(packed_main, selection.crop_token_indices)
    sub_ids = select_sparse_tokens(latent_ids, selection.crop_token_indices).clone()
    # Place the sparse sub branch immediately after the main target at t=20.
    sub_ids[..., 0] = 20
    return selection, sub_x0, sub_ids


def extract_sub_branch_batched(
    main_latents: torch.Tensor,
    selections: list[SparseTokenSelection],
    sub_t_coord: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract `sub_x0` and `sub_ids` for a batch using pre-computed `SparseTokenSelection`s.

    All `selections` must share the same `crop_h * crop_w` (enforced by BucketBatchSampler).
    Returns:
        sub_x0:  `[B, N_sub, D]` where `N_sub = crop_h * crop_w`.
        sub_ids: `[B, N_sub, 4]`; the `t_coord` column is overwritten to `sub_t_coord`.
    """
    device = main_latents.device
    batch_size = main_latents.shape[0]
    if len(selections) != batch_size:
        raise ValueError(
            f"`selections` length {len(selections)} must equal batch size {batch_size}."
        )

    packed_main = Flux2KleinPipeline._pack_latents(main_latents)
    latent_ids = Flux2KleinPipeline._prepare_latent_ids(main_latents).to(device=device)

    sub_x0_per_sample = []
    sub_ids_per_sample = []
    for b, sel in enumerate(selections):
        if sel.crop_token_indices is None or sel.crop_token_indices.numel() == 0:
            raise ValueError(f"Sample {b}: empty crop_token_indices in SparseTokenSelection.")
        if sel.main_token_indices is None or sel.main_token_indices.numel() == 0:
            raise ValueError(f"Sample {b}: empty main_token_indices in SparseTokenSelection.")
        crop_idx = sel.crop_token_indices.to(device=device, dtype=torch.long)
        sub_x0_per_sample.append(packed_main[b].index_select(0, crop_idx))
        sub_ids_b = latent_ids[b].index_select(0, crop_idx).clone()
        sub_ids_b[..., 0] = sub_t_coord
        sub_ids_per_sample.append(sub_ids_b)

    sub_x0 = torch.stack(sub_x0_per_sample, dim=0)
    sub_ids = torch.stack(sub_ids_per_sample, dim=0)
    return sub_x0, sub_ids


def move_selections_to_device(
    selections: list[SparseTokenSelection], device: torch.device
) -> list[SparseTokenSelection]:
    moved = []
    for sel in selections:
        moved.append(
            SparseTokenSelection(
                main_token_indices=sel.main_token_indices.to(device=device, dtype=torch.long),
                ref_token_indices=(
                    sel.ref_token_indices.to(device=device, dtype=torch.long)
                    if sel.ref_token_indices is not None
                    else None
                ),
                token_mask=sel.token_mask,
                token_coords=sel.token_coords,
                crop_token_indices=(
                    sel.crop_token_indices.to(device=device, dtype=torch.long)
                    if sel.crop_token_indices is not None
                    else None
                ),
                crop_token_coords=sel.crop_token_coords,
                crop_bounds=sel.crop_bounds,
            )
        )
    return moved


def get_sigmas(
    timesteps: torch.Tensor,
    noise_scheduler: FlowMatchEulerDiscreteScheduler,
    device: torch.device,
    n_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device=device)
    timesteps = timesteps.to(device=device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def build_dense_condition_ids(cond_latents_list: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    if not cond_latents_list:
        raise ValueError("Expected at least one condition latent.")

    cond_ids = []
    # Start conditions at t=40 and keep the same +20 spacing for later entries.
    for i, cond_latents in enumerate(cond_latents_list):
        latent_ids = Flux2KleinPipeline._prepare_latent_ids(cond_latents).to(device=device)
        latent_ids = latent_ids.clone()
        latent_ids[..., 0] = 40 + 20 * i
        cond_ids.append(latent_ids)

    return torch.cat(cond_ids, dim=1)


def build_transformer(
    args: argparse.Namespace,
    weight_dtype: torch.dtype,
    device: torch.device,
) -> torch.nn.Module:
    base_transformer = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        local_files_only=args.local_files_only,
    )

    if args.experiment_mode in {"no_sub", "sub_no_exchange"}:
        transformer = base_transformer
    else:
        config = base_transformer.config
        config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        valid_keys = set(inspect.signature(Flux2Transformer2DModel.__init__).parameters.keys()) - {"self"}
        init_kwargs = {key: value for key, value in config_dict.items() if key in valid_keys}
        pe_exchange_kwargs = {
            "dim_in": base_transformer.inner_dim,
            "pe_dim": sum(base_transformer.config.axes_dims_rope),
            "num_layers": len(base_transformer.transformer_blocks) + len(base_transformer.single_transformer_blocks),
            "sampler": "vanilla_ste",
            "encoder_layers": args.ste_encoder_layers,
            "encoder_num_heads": args.ste_encoder_num_heads,
            "head_init": args.ste_head_init,
        }
        custom_cls = (
            Flux2KleinPESoftExchangeTransformer2DModel
            if args.experiment_mode == "soft_exchange"
            else Flux2KleinPEExchangeTransformer2DModel
        )
        # (1) Build the whole custom transformer on META: the 9B DiT costs 0 bytes and every
        #     pe_exchange nn.init.* / register_buffer becomes a shape-only no-op. This removes
        #     the old fp32 second copy (~54GB/rank peak) that caused host OOM.
        with torch.device("meta"):
            transformer = custom_cls(pe_exchange_kwargs=pe_exchange_kwargs, **init_kwargs)

        # (2) Adopt base's bf16 DiT tensors BY REFERENCE (assign=True -> no copy, no fp32).
        missing, unexpected = transformer.load_state_dict(
            base_transformer.state_dict(), strict=False, assign=True
        )
        assert not unexpected, f"unexpected keys adopting base DiT: {unexpected[:8]}"
        non_pe_missing = [k for k in missing if not k.startswith("pe_exchange.")]
        assert not non_pe_missing, f"non-pe_exchange DiT keys left on meta: {non_pe_missing[:8]}"
        del base_transformer  # DiT storage now solely owned by `transformer` (~18GB bf16)

        # (3) Materialize pe_exchange via the SAME constructor path => identical init semantics
        #     (head_init zero/normal, STEBlock default-random init, tau=1.0 persistent buffer).
        #     type(...) auto-selects the hard/soft PackedSparsePEExchangeModel.
        ref_pe = type(transformer.pe_exchange)(**pe_exchange_kwargs)  # real CPU tensors, small
        transformer.pe_exchange.load_state_dict(ref_pe.state_dict(), assign=True, strict=True)
        del ref_pe

        # (4) Guarantee nothing survives on meta before the .to(cuda) at the end of this fn.
        for _n, _t in list(transformer.named_parameters()) + list(transformer.named_buffers()):
            assert not _t.is_meta, f"meta tensor survived: {_n}"

    transformer.requires_grad_(True)
    transformer.to(device=device, dtype=weight_dtype)
    if args.gradient_checkpointing:
        if getattr(args, "gc_use_reentrant", False):
            # Reentrant checkpoint traces differently under dynamo and avoids the
            # `lift_tracked_freevar_to_input` assertion that the diffusers default
            # (use_reentrant=False) hits when composed with torch.compile.
            import torch.utils.checkpoint as _torch_ckpt

            def _reentrant_gc_func(module, *gc_args):
                return _torch_ckpt.checkpoint(module.__call__, *gc_args, use_reentrant=True)

            transformer.enable_gradient_checkpointing(gradient_checkpointing_func=_reentrant_gc_func)
        else:
            transformer.enable_gradient_checkpointing()
    return transformer


def build_prompt_embeds(
    tokenizer: Qwen2TokenizerFast,
    text_encoder: Qwen3ForCausalLM,
    prompts: list[str],
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int,
    text_encoder_out_layers: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=prompts,
        device=device,
        dtype=dtype,
        max_sequence_length=max_sequence_length,
        hidden_states_layers=text_encoder_out_layers,
    )
    text_ids = Flux2KleinPipeline._prepare_text_ids(prompt_embeds).to(device=device)
    return prompt_embeds, text_ids


def build_optimizer(args: argparse.Namespace, params_to_optimize: list[dict[str, Any]]) -> torch.optim.Optimizer:
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    try:
        from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
    except ImportError as exc:
        raise ImportError(
            "Missing dependency `prodigyplus`. Install the same optimizer package used by Qwen before training."
        ) from exc

    kwargs = {
        "lr": args.learning_rate,
        "weight_decay": args.adam_weight_decay,
        "betas": (args.adam_beta1, args.adam_beta2),
        "d0": args.d0,
        "prodigy_steps": args.prodigy_steps,
        "use_schedulefree": True,
        "use_bias_correction": True,
        "safeguard_warmup": True,
    }
    signature = inspect.signature(ProdigyPlusScheduleFree.__init__)
    allowed = set(signature.parameters.keys())
    optimizer_kwargs = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    return ProdigyPlusScheduleFree(params_to_optimize, **optimizer_kwargs)


def unwrap_model(accelerator: Accelerator, model: torch.nn.Module) -> torch.nn.Module:
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    # accelerate.unwrap_model misses DDP when regional torch.compile is enabled
    # (the top-level wrapper is DDP, not OptimizedModule, so the compile path is
    # not taken; the DDP isn't peeled). Defensively unwrap.
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    if is_compiled_module(model):
        model = model._orig_mod
    return model


def save_checkpoint(
    accelerator: Accelerator,
    args: argparse.Namespace,
    global_step: int,
    checkpoints_total_limit: Optional[int],
    transformer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    output_dir = Path(args.output_dir)
    if checkpoints_total_limit is not None and accelerator.is_main_process:
        checkpoints = [path for path in output_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            for checkpoint in checkpoints[:num_to_remove]:
                shutil.rmtree(checkpoint)

    save_path = output_dir / f"checkpoint-{global_step}"
    if hasattr(optimizer, "eval"):
        optimizer.eval()
    accelerator.save_state(str(save_path))
    if hasattr(optimizer, "train"):
        optimizer.train()

    is_deepspeed = accelerator.state.distributed_type == DistributedType.DEEPSPEED
    if accelerator.is_main_process:
        # DS engine.save_checkpoint already wrote sharded fp32 master + bf16 weight under
        # `save_path`; an additional `save_pretrained` would race the DS sync path and could
        # capture the train-mode fp16 (schedule-free has shifted the fp32 master to eval view
        # but does not propagate back to forward fp16). Skip in DS mode.
        if not is_deepspeed:
            unwrap_model(accelerator, transformer).save_pretrained(save_path / "transformer")
        with (save_path / "training_state.json").open("w") as f:
            json.dump({"global_step": global_step, "is_deepspeed": is_deepspeed}, f, indent=2)


def maybe_resume(
    accelerator: Accelerator,
    args: argparse.Namespace,
    num_update_steps_per_epoch: int,
) -> tuple[int, int, int]:
    if not args.resume_from_checkpoint:
        return 0, 0, 0

    if args.resume_from_checkpoint != "latest":
        path = Path(args.resume_from_checkpoint)
    else:
        checkpoints = [p for p in Path(args.output_dir).iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
        if not checkpoints:
            logger.info("No checkpoint found for resume. Starting a new run.")
            return 0, 0, 0
        path = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))[-1]

    if not path.exists():
        logger.info("Resume checkpoint does not exist. Starting a new run.")
        return 0, 0, 0

    logger.info("Resuming from checkpoint %s", path)
    accelerator.load_state(str(path))
    global_step = int(path.name.split("-")[1])
    first_epoch = global_step // num_update_steps_per_epoch
    resume_step = (global_step % num_update_steps_per_epoch) * args.gradient_accumulation_steps
    return global_step, first_epoch, resume_step


def main() -> None:
    args = parse_args()
    if args.train_batch_size < 1:
        raise ValueError(f"train_batch_size must be >= 1, got {args.train_batch_size}.")
    if args.pe_exchange_region == "bbox" and args.use_sparse_sub_branch:
        raise ValueError(
            "--pe_exchange_region=bbox requires the dense sub branch; "
            "do not pass --use_sparse_sub_branch."
        )
    if args.pe_exchange_region == "bbox" and args.cache_dir is None:
        raise ValueError("--pe_exchange_region=bbox currently requires cache mode (--cache_dir).")

    # torch.compile stability for the dynamic-shape sparse workload (varying sub_seq_len
    # per step). Without these, the compiled gradient-checkpointed blocks recompile on
    # every new sequence length, and each recompile re-runs max-autotune, whose transient
    # benchmark workspace (~13 GB) stacks on top of the ~82 GB steady-state training memory
    # and OOMs by a hair. Goal: compile ONCE, reuse across all shapes.
    #   - capture_scalar_outputs: capture .item()/scalar reads instead of graph-breaking.
    #   - force_parameter_static_shapes=False: let one compiled graph serve the two
    #     modulation Linear weight shapes (double=24576 vs single=12288) — the exact
    #     dynamo guard that was firing — instead of recompiling per parameter shape.
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.force_parameter_static_shapes = False
    # Allow enough compiled-graph cache entries that a few legitimate variants
    # (double/single block, fwd/bwd) never silently fall back to eager.
    torch._dynamo.config.cache_size_limit = max(getattr(torch._dynamo.config, "cache_size_limit", 8), 64)
    torch._dynamo.config.accumulated_cache_size_limit = max(
        getattr(torch._dynamo.config, "accumulated_cache_size_limit", 256), 512
    )
    #   - suppress_errors: make torch.compile BEST-EFFORT. The gradient-checkpointed
    #     attention blocks, when a *second* graph variant is compiled (e.g. the
    #     subject-drop path or a new sub_seq_len), trip a hard dynamo assertion
    #     (`lift_tracked_freevar_to_input should not be called on root SubgraphTracer`)
    #     while tracing the checkpoint closure — this killed the process at step 2.
    #     A/B-verified on g00 (144 GB, so not OOM): use_reentrant True vs False makes
    #     NO difference — both crash identically. suppress_errors=True catches the
    #     failed compile and falls back to eager for that subgraph instead of dying,
    #     so max-autotune still compiles every graph it can and training never crashes.
    #     (No speed loss: compile gave no wall-clock speedup here anyway.)
    torch._dynamo.config.suppress_errors = True

    if args.experiment_mode == "no_sub" and args.sub_region_mode not in {"bbox", "mask"}:
        raise ValueError("Unexpected sub_region_mode value.")

    if args.zero_cond_t and args.experiment_mode != "hard_exchange":
        logger.warning(
            "--zero_cond_t was requested but --experiment_mode=%s does not use the patched "
            "Flux2KleinPEExchangeTransformer2DModel; the flag will be silently ignored.",
            args.experiment_mode,
        )

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=str(args.output_dir), logging_dir=str(logging_dir))
    # DDP memory optimizations needed to fit the torch.compile (max-autotune) backward,
    # whose fused buffers push peak per-GPU memory to ~95 GB — right at the 94.97 GB H100
    # limit. Two clean, no-DeepSpeed savings:
    #   - gradient_as_bucket_view=True: .grad fields are views into the DDP reduction
    #     buckets instead of separate allocations → eliminates a duplicate copy of the
    #     9B-param gradients (several GB saved).
    #   - find_unused_parameters=False: the eager run's reducer repeatedly reported "did
    #     not find any unused parameters" (hard_exchange always uses the STE gate + all
    #     DiT params), so this is safe and drops the extra graph traversal + its memory.
    kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )
    # Bump NCCL collective timeout from default 10 min to 60 min. Required because
    # a 9B model checkpoint (weights + optimizer + accelerate state) can take 5-15
    # minutes to flush to NFS — during which non-rank-0 workers idle on the next
    # collective and get killed by the watchdog if the timeout is too tight.
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=60))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs, pg_kwargs],
    )

    # DeepSpeed cannot infer the per-GPU micro batch size when the dataloader uses a
    # batch_sampler (BucketBatchSampler / PrecomputedBatchSampler → dataloader.batch_size is
    # None). Set it explicitly from --train_batch_size so ZeRO works with the bucketed loaders.
    # No-op for the plain DDP path (deepspeed_plugin is None).
    _ds_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if _ds_plugin is not None:
        _ds_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size
        # Optionally accumulate gradients in bf16 instead of DeepSpeed's fp32 default. The fp32
        # accumulation buffer is ~params*4B partitioned to this rank's optimizer shard (≈9GB/GPU
        # at 9B params / 4 GPUs) and is the difference between the bs=2 worst case fitting at
        # ~87GB (grad_accum=1, no buffer) and OOMing at ~96GB (grad_accum>1, fp32 buffer).
        if args.grad_accum_dtype is not None:
            _ds_plugin.deepspeed_config.setdefault("data_types", {})["grad_accum_dtype"] = args.grad_accum_dtype
            logger.info("DeepSpeed data_types.grad_accum_dtype = %s", args.grad_accum_dtype)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.report_to != "none":
        accelerator.init_trackers(
            project_name="flux2_klein_subject_sub",
            config=serialize_args(args),
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    cache_mode = args.cache_dir is not None

    tokenizer = None
    text_encoder = None
    vae = None
    latents_bn_mean = None
    latents_bn_std = None

    if not cache_mode:
        tokenizer = Qwen2TokenizerFast.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            local_files_only=args.local_files_only,
        )
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=weight_dtype,
            local_files_only=args.local_files_only,
        )
        text_encoder.requires_grad_(False)
        text_encoder.to(device=accelerator.device, dtype=weight_dtype)
        text_encoder.eval()

        vae = AutoencoderKLFlux2.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            torch_dtype=weight_dtype,
            local_files_only=args.local_files_only,
        )
        vae.requires_grad_(False)
        vae.to(device=accelerator.device, dtype=weight_dtype)
        vae.eval()

        latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device=accelerator.device, dtype=weight_dtype)
        latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
            device=accelerator.device,
            dtype=weight_dtype,
        )

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        local_files_only=args.local_files_only,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # A: staggered build to bound how many ranks sit in the CPU peak window at once.
    # BUILD_STAGGER_GROUP unset -> group == num_processes -> all ranks build together
    # (identical to the original behavior; no change to the g00 run).
    # BUILD_STAGGER_GROUP=k -> at most k ranks build concurrently.
    transformer = None
    _grp = int(os.environ.get("BUILD_STAGGER_GROUP", str(accelerator.num_processes)))
    _grp = max(1, min(_grp, accelerator.num_processes))
    _build_err = None
    for _start in range(0, accelerator.num_processes, _grp):
        if _start <= accelerator.process_index < _start + _grp:
            try:
                transformer = build_transformer(args, weight_dtype=weight_dtype, device=accelerator.device)
            except Exception as _e:  # keep non-failing ranks from hanging on the barrier
                _build_err = _e
                logger.error("rank %d build_transformer failed: %r", accelerator.process_index, _e)
        accelerator.wait_for_everyone()
    if _build_err is not None:
        raise _build_err
    assert transformer is not None, f"rank {accelerator.process_index} never built transformer"
    params_to_optimize = [{"params": list(transformer.parameters())}]
    optimizer = build_optimizer(args, params_to_optimize)

    if cache_mode:
        # NOTE: sparse sub branch at train_batch_size > 1 IS supported in cache mode — N_sparse
        # varies per sample, but collate_cached_examples pads the sub branch to the batch max and
        # emits `sub_valid_mask`, the batched PE swap (`_build_sparse_swapped_freqs_batched`)
        # consumes the per-sample selection list, the joint attention mask hides sub padding, and
        # `masked_branch_loss` drops padded positions. (Raw mode does NOT pad, so it still needs bs=1.)
        pass
        # Note: random per-sample long/short prompt (--long_prompt_prob) is safe at ANY
        # batch size in cache mode — the cache pads both `prompt_embeds` and
        # `long_prompt_embeds` to a fixed `max_sequence_length` (512), so mixed long/short
        # samples stack cleanly. (In raw mode the prompt is re-encoded per step and
        # --long_prompt_prob is ignored, since only the short `prompt` field is read.)
        train_dataset = CachedSubjectDrivenFlux2Dataset(
            dataset_path=args.dataset,
            cache_dir=args.cache_dir,
            max_train_samples=args.max_train_samples,
            precompute_bucket_keys=args.experiment_mode != "no_sub",
            use_long_prompt=args.use_long_prompt,
            long_prompt_prob=args.long_prompt_prob,
            use_sparse_sub_branch=args.use_sparse_sub_branch,
            pe_exchange_region=args.pe_exchange_region,
        )
        active_collate = collate_cached_examples
    else:
        train_dataset = SubjectDrivenFlux2Dataset(
            dataset_path=args.dataset,
            target_base=args.target_base,
            source_base=args.source_base,
            resolution=args.resolution,
            subject_image_field=args.subject_image_field,
            max_train_samples=args.max_train_samples,
            sub_region_mode=args.sub_region_mode,
            precompute_bucket_keys=args.experiment_mode != "no_sub",
        )
        active_collate = collate_examples

    # Sampler selection:
    #   - no_sub: plain DataLoader, no bucketing needed.
    #   - cache_mode + bucket_order_file: PrecomputedBatchSampler. The cached collate pads
    #     sub branches to batch max, so the precomputed batches no longer need uniform sizes
    #     (precompute_bucket_order.py now uses sort+greedy packing, no drop_last).
    #   - non-cache + bucket_order_file: PrecomputedBatchSampler with strict-bucket batches.
    #   - otherwise: dynamic BucketBatchSampler (strict bucket key, drop_last=True).
    if args.experiment_mode == "no_sub":
        if cache_mode and args.bucket_order_file is not None and args.train_batch_size > 1:
            # Native-aspect cache: collate_cached_examples hard-`torch.stack`s packed_target /
            # packed_main / cond_image_ids, so batches MUST be main-resolution-homogeneous at
            # bs>1 (a plain shuffled DataLoader would crash the stack). Reuse the same
            # main-resolution-grouped bucket order as the sub-branch modes — the sub-area
            # secondary sort is a harmless no-op for no_sub (no sub branch).
            precomputed = load_precomputed_batches(
                resolve_path(args.bucket_order_file),
                dataset_size=len(train_dataset),
                expected_batch_size=args.train_batch_size,
            )
            logger.info(
                "no_sub bs=%d: using %d precomputed main-grouped batches from %s.",
                args.train_batch_size, len(precomputed), args.bucket_order_file,
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_sampler=PrecomputedBatchSampler(precomputed),
                collate_fn=active_collate,
                num_workers=args.dataloader_num_workers,
            )
        else:
            train_dataloader = DataLoader(
                train_dataset,
                shuffle=True,
                collate_fn=active_collate,
                batch_size=args.train_batch_size,
                num_workers=args.dataloader_num_workers,
            )
    elif args.bucket_order_file is not None:
        precomputed = load_precomputed_batches(
            resolve_path(args.bucket_order_file),
            dataset_size=len(train_dataset),
            expected_batch_size=args.train_batch_size,
        )
        logger.info(
            "Loaded %d precomputed batches from %s (train_batch_size=%d).",
            len(precomputed),
            args.bucket_order_file,
            args.train_batch_size,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=PrecomputedBatchSampler(precomputed),
            collate_fn=active_collate,
            num_workers=args.dataloader_num_workers,
        )
    else:
        bucket_sampler = BucketBatchSampler(
            train_dataset,
            batch_size=args.train_batch_size,
            drop_last=True,
            shuffle=True,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=bucket_sampler,
            collate_fn=active_collate,
            num_workers=args.dataloader_num_workers,
        )

    transformer, optimizer, train_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader)
    if hasattr(optimizer, "train"):
        optimizer.train()

    # torch.compile under DeepSpeed: accelerate's auto dynamo path (engine.compile via the yaml
    # dynamo_config) did NOT actually fire max-autotune in practice (eager step times, 0 autotune).
    # Trigger it explicitly here. ZeRO-1 keeps params/grads replicated (only optimizer state is
    # sharded), so the forward graph is DDP-like and compiles the same way it does under plain DDP.
    # Idempotent: DeepSpeedEngine.compile() early-returns if already compiled. Gated by
    # --deepspeed_compile so the plain-DDP path (accelerate's own dynamo) is untouched.
    if _ds_plugin is not None and getattr(args, "deepspeed_compile", False) and hasattr(transformer, "compile"):
        transformer.compile(
            backend="inductor",
            compile_kwargs={"mode": "max-autotune-no-cudagraphs", "dynamic": True, "fullgraph": False},
        )
        logger.info("DeepSpeed engine torch.compile fired (max-autotune-no-cudagraphs).")

    text_layers = parse_text_layers(args.text_encoder_out_layers)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    global_step, first_epoch, resume_step = maybe_resume(accelerator, args, num_update_steps_per_epoch)
    # accelerate.load_state restores the optimizer state including ScheduleFree's
    # internal train_mode flag. Since save_checkpoint stores while in eval mode
    # (required for save_state to capture the right fp32 master view), the resumed
    # optimizer comes back in eval mode and the next step() would raise
    # "Not in train mode!". Flip it back to train.
    if hasattr(optimizer, "train"):
        optimizer.train()

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Steps",
    )

    def branch_loss(pred: torch.Tensor, noise: torch.Tensor, x0: torch.Tensor, weighting: torch.Tensor) -> torch.Tensor:
        target = noise - x0
        loss = torch.mean((weighting.float() * (pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), dim=1)
        return loss.mean()

    def masked_branch_loss(
        pred: torch.Tensor,
        noise: torch.Tensor,
        x0: torch.Tensor,
        weighting: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Branch loss restricted to valid (non-padding) sub-token positions.

        `valid_mask` is `[B, N_sub]` bool (True at real tokens). When None, falls back to
        the plain branch_loss (matches existing strict-bucket behavior).
        """
        if valid_mask is None:
            return branch_loss(pred, noise, x0, weighting)
        target = noise - x0
        sq = (weighting.float() * (pred.float() - target.float()) ** 2).sum(dim=-1)  # [B, N_sub]
        mask = valid_mask.to(dtype=sq.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        per_sample = (sq * mask).sum(dim=1) / denom
        # Divide by feature dim to match the un-summed reduce in branch_loss.
        per_sample = per_sample / pred.shape[-1]
        return per_sample.mean()

    # CFG prompt-dropout NULL embedding: the encoded empty prompt ("") = the same unconditional
    # branch Klein-base inference uses (negative_prompt=""). Loaded once and reused for the dropped
    # samples so prompt-dropout training matches two-pass text-CFG inference exactly.
    null_prompt_embeds = None
    null_text_ids = None
    if args.prompt_drop_prob and args.prompt_drop_prob > 0.0:
        if not cache_mode:
            raise ValueError("--prompt_drop_prob is cache-mode only (needs the precomputed null embedding).")
        null_path = args.null_prompt_embeds_path
        if null_path is None:
            null_path = str(resolve_path(args.cache_dir) / "_null_prompt_embeds.safetensors")
        if not Path(null_path).exists():
            raise FileNotFoundError(
                f"--prompt_drop_prob={args.prompt_drop_prob} needs the null prompt embedding at {null_path}. "
                f"Generate it: python precompute_null_prompt.py --cache_dir {args.cache_dir}"
            )
        _null = safetensors_load_file(null_path)
        null_prompt_embeds = _null["null_prompt_embeds"].to(device=accelerator.device, dtype=weight_dtype)
        null_text_ids = _null["null_text_ids"].to(device=accelerator.device)
        if null_prompt_embeds.ndim == 3:  # stored as [1, S, D] -> [S, D]
            null_prompt_embeds = null_prompt_embeds.squeeze(0)
        if null_text_ids.ndim == 3:
            null_text_ids = null_text_ids.squeeze(0)
        logger.info("CFG prompt dropout enabled: p=%.3f, null embed %s from %s",
                    args.prompt_drop_prob, tuple(null_prompt_embeds.shape), null_path)

    last_sync_t = time.perf_counter()
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(transformer):
                is_cached_batch = bool(batch.get("is_cached", False))
                prompts = batch["prompts"]
                sub_valid_mask = batch.get("sub_valid_mask")  # cache mode only
                sub_has_padding = bool(batch.get("sub_has_padding", False))

                if is_cached_batch:
                    # All heavy encoding skipped — load pre-cached tensors.
                    prompt_embeds = batch["prompt_embeds"].to(device=accelerator.device, dtype=weight_dtype)
                    text_ids = batch["text_ids"].to(device=accelerator.device)
                    packed_target = batch["packed_target"].to(device=accelerator.device, dtype=weight_dtype)
                    packed_main = batch["packed_main"].to(device=accelerator.device, dtype=weight_dtype)
                    packed_subject = batch["packed_subject"].to(device=accelerator.device, dtype=weight_dtype)
                    model_input_ids = batch["target_latent_ids"].to(device=accelerator.device)
                    cond_image_ids = batch["cond_image_ids"].to(device=accelerator.device)
                    sub_x0_cached = batch["subcrop_packed"].to(device=accelerator.device, dtype=weight_dtype)
                    sub_ids_cached = batch["subcrop_latent_ids"].to(device=accelerator.device)
                    if sub_valid_mask is not None:
                        sub_valid_mask = sub_valid_mask.to(device=accelerator.device)
                    main_latents = None  # unused in cache path
                else:
                    with torch.no_grad():
                        prompt_embeds, text_ids = build_prompt_embeds(
                            tokenizer=tokenizer,
                            text_encoder=text_encoder,
                            prompts=prompts,
                            device=accelerator.device,
                            dtype=weight_dtype,
                            max_sequence_length=args.max_sequence_length,
                            text_encoder_out_layers=text_layers,
                        )
                        target_pixels = batch["target_pixels"].to(device=accelerator.device, dtype=weight_dtype)
                        main_pixels = batch["main_pixels"].to(device=accelerator.device, dtype=weight_dtype)
                        subject_pixels = batch["subject_pixels"].to(device=accelerator.device, dtype=weight_dtype)
                        target_latents = encode_latents(vae, target_pixels, latents_bn_mean, latents_bn_std)
                        main_latents = encode_latents(vae, main_pixels, latents_bn_mean, latents_bn_std)
                        subject_latents = encode_latents(vae, subject_pixels, latents_bn_mean, latents_bn_std)
                    packed_target = Flux2KleinPipeline._pack_latents(target_latents)
                    packed_main = Flux2KleinPipeline._pack_latents(main_latents)
                    packed_subject = Flux2KleinPipeline._pack_latents(subject_latents)
                    model_input_ids = Flux2KleinPipeline._prepare_latent_ids(target_latents).to(device=accelerator.device)
                    cond_image_ids_single = build_dense_condition_ids(
                        [main_latents[0:1], subject_latents[0:1]],
                        device=accelerator.device,
                    )
                    cond_image_ids = cond_image_ids_single.expand(packed_target.shape[0], -1, -1)

                bsz = packed_target.shape[0]
                selections = batch.get("selections", None)
                if selections is not None:
                    selections = move_selections_to_device(selections, accelerator.device)

                # CFG TEXT dropout (PER SAMPLE): replace the prompt with the encoded empty ("") NULL
                # embedding so the model learns the SAME unconditional branch Klein-base inference uses
                # (two-pass text CFG, negative_prompt=""). All image conditions (subject reference,
                # background, sub branch, PE exchange) are left untouched — only the text is dropped.
                # Per-sample (not per-batch) so the realised drop rate matches `prompt_drop_prob` even
                # at bs>1. Seeded on (seed, epoch, step, rank) for reproducibility; per-rank so ranks
                # drop independent samples. No collective depends on this, so no sync needed.
                n_prompt_dropped = 0
                if null_prompt_embeds is not None:
                    _ptxt_rng = random.Random(
                        args.seed * 7919 + epoch * 104729 + step * 131 + accelerator.process_index
                    )
                    drop_mask = torch.tensor(
                        [_ptxt_rng.random() < args.prompt_drop_prob for _ in range(bsz)],
                        device=accelerator.device,
                    )
                    if bool(drop_mask.any()):
                        prompt_embeds = prompt_embeds.clone()
                        text_ids = text_ids.clone()
                        _idx = drop_mask.nonzero(as_tuple=False).squeeze(-1)
                        prompt_embeds[_idx] = null_prompt_embeds.to(prompt_embeds.dtype)
                        text_ids[_idx] = null_text_ids.to(text_ids.dtype)
                        n_prompt_dropped = int(drop_mask.sum().item())

                # Subject-IMAGE dropout: with prob `subject_drop_prob` the SUBJECT REFERENCE IMAGE
                # condition (`packed_subject`, the cond latent at t=60) is removed from packed_cond
                # for this step. The sub branch and PE exchange are NOT dropped — they operate on the
                # target/background tokens (not on `packed_subject`), so they stay ON every step.
                # Decided PER BATCH (one roll per micro-step) so packed_cond keeps a uniform sequence
                # length within the batch (needed to stack at train_batch_size > 1). Seeding on
                # (seed, epoch, step) keeps it deterministic and reproducible across resumes.
                _drop_roll = random.Random(
                    args.seed * 1_000_003 + epoch * 100_003 + step
                ).random()
                drop_subject = (
                    args.experiment_mode != "no_sub"
                    and args.subject_drop_prob > 0.0
                    and _drop_roll < args.subject_drop_prob
                )
                # Sub branch + PE exchange are KEPT regardless of drop_subject (only the reference
                # image cond is dropped). This also means the STE/PE params are used on EVERY step,
                # so no DDP unused-param / ZeRO-3 collective-divergence issue and no zero-touch needed.
                use_sub_branch = (args.experiment_mode != "no_sub")

                if drop_subject:
                    # packed_cond / cond_image_ids hold [background | subject]; keep background.
                    main_cond_len = packed_main.shape[1]
                    packed_cond = packed_main
                    cond_image_ids = cond_image_ids[:, :main_cond_len, :]
                else:
                    packed_cond = torch.cat([packed_main, packed_subject], dim=1)

                noise = torch.randn_like(packed_target)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=packed_target.device)
                sigmas = get_sigmas(
                    timesteps,
                    noise_scheduler=noise_scheduler_copy,
                    device=accelerator.device,
                    n_dim=packed_target.ndim,
                    dtype=packed_target.dtype,
                )
                noisy_target = (1.0 - sigmas) * packed_target + sigmas * noise

                hidden_states = torch.cat([noisy_target, packed_cond], dim=1)
                img_ids = torch.cat([model_input_ids, cond_image_ids], dim=1)
                pe_sparse_token_selection = None
                sub_noise = None
                sub_x0 = None
                sub_seq_len = 0
                sub_t_coord = 0

                if use_sub_branch:
                    if selections is None:
                        raise RuntimeError(
                            "Batch is missing precomputed `selections`. "
                            "Did you forget `precompute_bucket_keys=True` / `sub_region_mode` in the Dataset?"
                        )
                    if is_cached_batch:
                        sub_x0 = sub_x0_cached
                        sub_ids = sub_ids_cached
                    else:
                        sub_x0, sub_ids = extract_sub_branch_batched(
                            main_latents=main_latents,
                            selections=selections,
                            sub_t_coord=20,
                        )
                    sub_noise = torch.randn_like(sub_x0)
                    sub_sigmas = get_sigmas(
                        timesteps,
                        noise_scheduler=noise_scheduler_copy,
                        device=accelerator.device,
                        n_dim=sub_x0.ndim,
                        dtype=sub_x0.dtype,
                    )
                    noisy_sub = (1.0 - sub_sigmas) * sub_x0 + sub_sigmas * sub_noise
                    hidden_states = torch.cat([noisy_target, noisy_sub, packed_cond], dim=1)
                    img_ids = torch.cat([model_input_ids, sub_ids, cond_image_ids], dim=1)
                    sub_seq_len = sub_x0.shape[1]
                    sub_t_coord = int(sub_ids[0, 0, 0].item())

                    if args.experiment_mode in {"soft_exchange", "hard_exchange"}:
                        pe_sparse_token_selection = selections

                if unwrap_model(accelerator, transformer).config.guidance_embeds:
                    guidance = torch.full([1], args.guidance_scale, device=accelerator.device, dtype=packed_target.dtype)
                    guidance = guidance.expand(bsz)
                else:
                    guidance = None

                # Build attention mask for the joint (text + image) attention. The text
                # segment is always full (no padding); image segment is
                # [noisy_target | noisy_sub | packed_main | packed_subject]; only the
                # noisy_sub segment may contain padding (cached mode with mixed sub sizes).
                joint_attention_kwargs: dict[str, Any] = {}
                image_valid_mask = None
                pe_ste_key_padding_mask = None
                if sub_valid_mask is not None and sub_seq_len > 0:
                    text_seq = prompt_embeds.shape[1]
                    main_seq_len_for_mask = packed_target.shape[1]
                    cond_seq = packed_cond.shape[1]
                    image_valid_mask = torch.cat(
                        [
                            torch.ones(
                                (bsz, main_seq_len_for_mask),
                                dtype=torch.bool,
                                device=accelerator.device,
                            ),
                            sub_valid_mask,
                            torch.ones((bsz, cond_seq), dtype=torch.bool, device=accelerator.device),
                        ],
                        dim=1,
                    )
                    text_valid_mask = torch.ones(
                        (bsz, text_seq), dtype=torch.bool, device=accelerator.device
                    )
                    attn_mask = torch.cat([text_valid_mask, image_valid_mask], dim=1)
                    # SDPA broadcasts [B, 1, 1, L_kv] across heads and queries.
                    joint_attention_kwargs["attention_mask"] = attn_mask[:, None, None, :]

                model_kwargs: dict[str, Any] = {
                    "hidden_states": hidden_states,
                    "timestep": timesteps / 1000,
                    "guidance": guidance,
                    "encoder_hidden_states": prompt_embeds,
                    "txt_ids": text_ids,
                    "img_ids": img_ids,
                    "return_dict": False,
                }
                if joint_attention_kwargs:
                    model_kwargs["joint_attention_kwargs"] = joint_attention_kwargs
                if pe_sparse_token_selection is not None:
                    model_kwargs["pe_sparse_token_selection"] = pe_sparse_token_selection
                    model_kwargs["pe_ref_t_coord"] = sub_t_coord
                    model_kwargs["pe_main_t_coord"] = 0
                    if image_valid_mask is not None and sub_has_padding:
                        # Training/collate uses True=real, while TransformerEncoder uses
                        # True=ignore. Invert exactly once at the STE API boundary. Keep
                        # None for no-padding batches so their historical execution path is
                        # unchanged.
                        pe_ste_key_padding_mask = ~image_valid_mask
                        model_kwargs["pe_ste_key_padding_mask"] = pe_ste_key_padding_mask
                # `num_cond_tokens` is only accepted by Flux2KleinPEExchangeTransformer2DModel
                # (the hard-exchange variant patched in this repo). The base
                # Flux2Transformer2DModel (used by no_sub / sub_no_exchange) and the soft
                # variant don't accept the kwarg, so silently skip there.
                if args.zero_cond_t and args.experiment_mode == "hard_exchange":
                    model_kwargs["num_cond_tokens"] = int(packed_cond.shape[1])

                # Mark the image sequence axis (dim 1) dynamic so torch.compile builds ONE
                # graph that serves every sub_seq_len, instead of recompiling (and re-running
                # max-autotune) each step a new sparse-token count appears. The total image
                # sequence length = target + sub + cond; only the sub segment varies, but the
                # concatenated length changes, so the whole dim is dynamic. Idempotent per step.
                torch._dynamo.mark_dynamic(hidden_states, 1)
                torch._dynamo.mark_dynamic(img_ids, 1)
                if pe_ste_key_padding_mask is not None:
                    torch._dynamo.mark_dynamic(pe_ste_key_padding_mask, 1)

                model_pred = transformer(**model_kwargs)[0]
                main_seq_len = packed_target.shape[1]
                main_pred = model_pred[:, :main_seq_len, :]

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                main_loss = branch_loss(main_pred, noise, packed_target, weighting)

                if sub_x0 is not None and sub_noise is not None:
                    sub_pred = model_pred[:, main_seq_len : main_seq_len + sub_seq_len, :]
                    sub_weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sub_sigmas)
                    sub_loss = masked_branch_loss(sub_pred, sub_noise, sub_x0, sub_weighting, sub_valid_mask)
                    loss = (main_loss + args.sub_loss_weight * sub_loss) / (1.0 + args.sub_loss_weight)
                else:
                    sub_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)
                    loss = main_loss

                # (No zero-touch needed: the sub branch + PE exchange / STE params now run on EVERY
                # step — subject dropout only removes the `packed_subject` cond — so every trainable
                # param is always in the autograd graph. DDP find_unused_parameters=False and all
                # ZeRO stages are satisfied without the previous zero-weighted param-sum hack.)

                accelerator.backward(loss)
                grad_norm_val: Optional[float] = None
                if accelerator.sync_gradients:
                    _gn = accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                    if _gn is not None:
                        try:
                            grad_norm_val = float(_gn.detach().item() if torch.is_tensor(_gn) else _gn)
                        except Exception:
                            grad_norm_val = None
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                now_t = time.perf_counter()
                step_time_s = now_t - last_sync_t
                last_sync_t = now_t
                effective_bsz = bsz * args.gradient_accumulation_steps * accelerator.num_processes
                samples_per_sec = effective_bsz / step_time_s if step_time_s > 0 else 0.0

                logs = {
                    "loss": loss.detach().item(),
                    "main_loss": main_loss.detach().item(),
                    "sub_loss": sub_loss.detach().item(),
                    "selected_sub_tokens": 0 if sub_x0 is None else int(sub_x0.shape[1]),
                    # Optimizer / training dynamics
                    "epoch_frac": global_step / max(num_update_steps_per_epoch, 1),
                    # Diffusion noise schedule monitoring
                    "train/timestep_mean": float(timesteps.float().mean().item()),
                    "train/timestep_min": float(timesteps.float().min().item()),
                    "train/timestep_max": float(timesteps.float().max().item()),
                    "train/sigma_mean": float(sigmas.float().mean().item()),
                    # Sequence length / padding stats
                    "seq/sub_seq_len_padded": int(sub_seq_len),
                    "seq/main_seq_len": int(packed_target.shape[1]),
                    "seq/total_image_seq_len": int(hidden_states.shape[1]),
                    # Condition / prompt dropout monitoring (this step's decision)
                    "train/subject_dropped": 1.0 if drop_subject else 0.0,
                    "train/prompt_dropped_frac": (n_prompt_dropped / max(bsz, 1)),
                    # Performance / throughput
                    "perf/step_time_s": step_time_s,
                    "perf/samples_per_sec": samples_per_sec,
                }
                if grad_norm_val is not None:
                    logs["grad_norm"] = grad_norm_val
                # Long-vs-short prompt mix (cache mode; rate of long prompts in this batch)
                used_long_prompts = batch.get("used_long_prompts")
                if used_long_prompts:
                    logs["train/used_long_prompt_rate"] = sum(
                        1.0 for x in used_long_prompts if x
                    ) / len(used_long_prompts)
                # Sub-branch valid-token / padding stats
                if sub_valid_mask is not None and sub_seq_len > 0:
                    valid = float(sub_valid_mask.float().sum().item())
                    total_sub = float(sub_valid_mask.numel())
                    logs["seq/sub_valid_tokens_mean"] = valid / max(sub_valid_mask.shape[0], 1)
                    logs["seq/sub_padding_ratio"] = 1.0 - (valid / max(total_sub, 1.0))
                # ProdigyPlusScheduleFree internal state (adaptive step-size)
                pg0 = optimizer.param_groups[0]
                if "d" in pg0:
                    logs["prodigy/d"] = float(pg0.get("d", 0.0))
                    logs["prodigy/effective_lr"] = float(pg0.get("effective_lr", 0.0))
                    logs["prodigy/k"] = int(pg0.get("k", 0))
                # GPU memory (local rank; representative when load is balanced)
                if torch.cuda.is_available():
                    dev = accelerator.device
                    logs["perf/gpu_mem_alloc_gb"] = torch.cuda.memory_allocated(dev) / (1024 ** 3)
                    logs["perf/gpu_mem_reserved_gb"] = torch.cuda.memory_reserved(dev) / (1024 ** 3)
                    logs["perf/gpu_mem_max_alloc_gb"] = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
                    torch.cuda.reset_peak_memory_stats(dev)

                progress_bar.set_postfix({k: v for k, v in logs.items() if k in ("loss", "main_loss", "sub_loss", "grad_norm")})
                accelerator.log(logs, step=global_step)

                if global_step % args.checkpointing_steps == 0:
                    save_checkpoint(
                        accelerator=accelerator,
                        args=args,
                        global_step=global_step,
                        checkpoints_total_limit=args.checkpoints_total_limit,
                        transformer=transformer,
                        optimizer=optimizer,
                    )

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    is_deepspeed = accelerator.state.distributed_type == DistributedType.DEEPSPEED
    if accelerator.is_main_process:
        # In DS mode, the final transformer/ snapshot is omitted for the same reason as
        # per-step checkpoints (see save_checkpoint). To export a fp16 HuggingFace snapshot
        # from a DS run, use the per-step DS checkpoint with `zero_to_fp32.py` (auto-emitted
        # by DS under each `checkpoint-*/`) and load the resulting consolidated weights.
        if not is_deepspeed:
            final_dir = Path(args.output_dir) / "final_transformer"
            unwrap_model(accelerator, transformer).save_pretrained(final_dir)
        with (Path(args.output_dir) / "train_args.json").open("w") as f:
            json.dump(serialize_args(args), f, indent=2)

    accelerator.end_training()


if __name__ == "__main__":
    main()
