from __future__ import annotations

import argparse
import copy
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parent
LOCAL_DIFFUSERS_SRC = REPO_ROOT / "diffusers" / "src"
if LOCAL_DIFFUSERS_SRC.exists() and str(LOCAL_DIFFUSERS_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIFFUSERS_SRC))

from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel  # noqa: E402
from diffusers.pipelines.flux2.pipeline_flux2_klein import (  # noqa: E402
    Flux2PipelineOutput,
    compute_empirical_mu,
    retrieve_timesteps,
)
from diffusers.utils.torch_utils import randn_tensor  # noqa: E402

from flux2_klein_pe_exchange import Flux2KleinPEExchangeTransformer2DModel  # noqa: E402
from flux2_klein_pe_soft_exchange import Flux2KleinPESoftExchangeTransformer2DModel  # noqa: E402
from qwen_pe_exchange_sparse_model import (  # noqa: E402
    SparseTokenSelection,
    build_bbox_pe_exchange_selection,
    build_sparse_token_selection_from_mask,
    select_sparse_tokens,
)
from qwen_pe_exchange_sparse_soft_model import SparseTokenSelection as SoftSparseTokenSelection  # noqa: E402


DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-base-9B"
VALID_EXPERIMENT_MODES = {"no_sub", "sub_no_exchange", "soft_exchange", "hard_exchange"}
VALID_SEGMENTS = {"target", "cond", "sub"}


@dataclass(frozen=True)
class LayoutConfig:
    timestep_layout: str
    hidden_state_order: tuple[str, ...]
    img_ids_order: tuple[str, ...]
    sub_t_coord: int
    condition_t_start: int
    condition_t_stride: int = 20


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def parse_text_layers(spec: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in spec.split(",") if x.strip())


def to_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype {name}.")


def normalize_order(order: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(order, str):
        parts = tuple(part.strip() for part in order.split(",") if part.strip())
    else:
        parts = tuple(order)
    if not parts:
        raise ValueError("Token order must not be empty.")
    if parts[0] != "target":
        raise ValueError(f"Token order must start with `target`, got {parts}.")
    if set(parts) != VALID_SEGMENTS:
        raise ValueError(
            f"Token order must be a permutation of {sorted(VALID_SEGMENTS)}, got {parts}."
        )
    if len(parts) != len(set(parts)):
        raise ValueError(f"Token order contains duplicates: {parts}.")
    return parts


def filter_order_for_mode(order: tuple[str, ...], has_sub: bool) -> tuple[str, ...]:
    return tuple(segment for segment in order if has_sub or segment != "sub")


def resolve_layout(
    *,
    experiment_mode: str,
    layout: str = "auto",
    hidden_state_order: str | tuple[str, ...] | list[str] | None = None,
    img_ids_order: str | tuple[str, ...] | list[str] | None = None,
    sub_t_coord: Optional[int] = None,
    condition_t_start: Optional[int] = None,
    condition_t_stride: int = 20,
) -> LayoutConfig:
    if experiment_mode not in VALID_EXPERIMENT_MODES:
        raise ValueError(f"Unsupported experiment_mode {experiment_mode}.")

    if layout == "auto":
        layout = "ids20" if experiment_mode == "hard_exchange" else "t60"
    if layout not in {"t60", "ids20"}:
        raise ValueError(f"Unsupported layout {layout}.")

    # `layout` is only a t-coordinate preset. Token order and img_ids order remain
    # independent knobs even if we provide matching convenience defaults here.
    if layout == "ids20":
        default_hidden = ("target", "sub", "cond")
        default_ids = ("target", "sub", "cond")
        default_sub_t = 20
        default_cond_t = 40
    else:
        default_hidden = ("target", "cond", "sub")
        default_ids = ("target", "cond", "sub")
        default_sub_t = 60
        default_cond_t = 20

    return LayoutConfig(
        timestep_layout=layout,
        hidden_state_order=normalize_order(hidden_state_order or default_hidden),
        img_ids_order=normalize_order(img_ids_order or default_ids),
        sub_t_coord=default_sub_t if sub_t_coord is None else int(sub_t_coord),
        condition_t_start=default_cond_t if condition_t_start is None else int(condition_t_start),
        condition_t_stride=int(condition_t_stride),
    )


def _device_string(device: torch.device) -> str:
    if device.type == "cuda" and device.index is not None:
        return f"cuda:{device.index}"
    return device.type


def _materialize_pipeline_modules(
    pipe: Flux2KleinPipeline,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Flux2KleinPipeline:
    pipe.transformer = pipe.transformer.to(device=device, dtype=dtype)
    pipe.vae = pipe.vae.to(device=device, dtype=dtype)
    pipe.text_encoder = pipe.text_encoder.to(device=device, dtype=dtype)
    return pipe


def load_pil_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image)
    return ImageOps.exif_transpose(Image.open(resolve_path(image)))


def image_to_tensor(
    image: str | Path | Image.Image,
    *,
    width: int,
    height: int,
) -> torch.Tensor:
    pil = load_pil_image(image).convert("RGB")
    pil = pil.resize((width, height), Image.Resampling.BILINEAR)
    tensor = torch.frombuffer(bytearray(pil.tobytes()), dtype=torch.uint8).view(height, width, 3)
    tensor = tensor.permute(2, 0, 1).to(torch.float32).div_(255.0)
    return tensor.mul_(2.0).sub_(1.0).unsqueeze(0).contiguous()


def mask_to_tensor(
    mask: str | Path | Image.Image | torch.Tensor,
    *,
    width: int,
    height: int,
) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        mask_tensor = mask.to(torch.float32)
        if mask_tensor.dim() == 4:
            mask_tensor = mask_tensor[0, 0]
        elif mask_tensor.dim() == 3:
            mask_tensor = mask_tensor[0]
        if float(mask_tensor.max().item()) > 1.0:
            mask_tensor = (mask_tensor > 128.0).to(torch.float32)
        else:
            mask_tensor = (mask_tensor > 0.5).to(torch.float32)
        if tuple(mask_tensor.shape[-2:]) != (height, width):
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0),
                size=(height, width),
                mode="nearest",
            )[0, 0]
        return mask_tensor.contiguous()

    pil = load_pil_image(mask).convert("L")
    pil = pil.resize((width, height), Image.Resampling.NEAREST)
    tensor = torch.frombuffer(bytearray(pil.tobytes()), dtype=torch.uint8).view(height, width).to(torch.float32)
    return (tensor > 128.0).to(torch.float32).contiguous()


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


def build_condition_ids(
    condition_latents: list[torch.Tensor],
    *,
    start_t: int,
    stride: int,
    device: torch.device,
) -> torch.Tensor:
    if not condition_latents:
        raise ValueError("Expected at least one condition latent.")

    cond_ids = []
    for index, latents in enumerate(condition_latents):
        latent_ids = Flux2KleinPipeline._prepare_latent_ids(latents).to(device=device).clone()
        latent_ids[..., 0] = start_t + stride * index
        cond_ids.append(latent_ids)
    return torch.cat(cond_ids, dim=1)


def build_sparse_sub_branch(
    main_latents: torch.Tensor,
    mask: torch.Tensor,
    *,
    sub_region_mode: str,
    sub_t_coord: int,
    use_sparse_sub_branch: bool = True,
    pe_exchange_region: str = "auto",
    generator: torch.Generator | list[torch.Generator] | None = None,
    batch_size: int | None = None,
) -> tuple[SparseTokenSelection, torch.Tensor, torch.Tensor]:
    if pe_exchange_region not in {"auto", "mask", "bbox"}:
        raise ValueError(f"Unsupported pe_exchange_region {pe_exchange_region}.")

    mask_selection = build_sparse_token_selection_from_mask(mask, downsample_factor=16, threshold=0.5)
    if sub_region_mode == "mask":
        selection = mask_selection
    elif sub_region_mode == "bbox":
        selection = build_bbox_token_selection_from_mask(mask, downsample_factor=16, threshold=0.5)
    else:
        raise ValueError(f"Unsupported sub_region_mode {sub_region_mode}.")

    if selection.main_token_indices.numel() == 0:
        raise ValueError("The selected mask/bbox region produced zero sub tokens.")
    if selection.crop_token_indices is None or selection.crop_token_indices.numel() == 0:
        raise ValueError("The selected mask/bbox region produced zero crop tokens.")
    if selection.crop_bounds is None or selection.ref_token_indices is None:
        raise ValueError("The selected mask/bbox region produced no bbox-local token mapping.")

    y1, y2, x1, x2 = selection.crop_bounds
    crop_h = int(y2 - y1)
    crop_w = int(x2 - x1)
    effective_batch_size = main_latents.shape[0] if batch_size is None else int(batch_size)
    if effective_batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {effective_batch_size}.")

    # Match sparse-sub generation to the training target geometry: initialize a dense
    # bbox-local latent grid from Gaussian noise, then retain only the bbox-local mask
    # tokens. Unlike the old path, no token value is copied from the main/background.
    dense_sub_latents = randn_tensor(
        (effective_batch_size, main_latents.shape[1], crop_h, crop_w),
        generator=generator,
        device=main_latents.device,
        dtype=main_latents.dtype,
    )
    packed_sub = Flux2KleinPipeline._pack_latents(dense_sub_latents)
    dense_sub_ids = Flux2KleinPipeline._prepare_latent_ids(dense_sub_latents).to(device=main_latents.device)
    dense_sub_ids[..., 0] = int(sub_t_coord)

    if use_sparse_sub_branch:
        if pe_exchange_region == "bbox" and sub_region_mode != "bbox":
            raise ValueError(
                "bbox-wide PE exchange needs all bbox-local sub tokens; use the dense sub branch "
                "or sub_region_mode='bbox'."
            )
        ref_idx = selection.ref_token_indices.to(device=main_latents.device, dtype=torch.long)
        sub_latents = select_sparse_tokens(packed_sub, ref_idx)
        sub_ids = select_sparse_tokens(dense_sub_ids, ref_idx).clone()
        selection.ref_token_indices = torch.arange(
            ref_idx.shape[0], device=main_latents.device, dtype=torch.long
        )
        if pe_exchange_region == "mask" and sub_region_mode == "bbox":
            # The sub sequence is the full bbox, but PE exchange is explicitly mask-only.
            selection = mask_selection
        return selection, sub_latents, sub_ids

    resolved_pe_region = sub_region_mode if pe_exchange_region == "auto" else pe_exchange_region
    if resolved_pe_region == "mask":
        selection = mask_selection
    else:
        selection = build_bbox_pe_exchange_selection(mask_selection, packed_sub.shape[1])
    return selection, packed_sub, dense_sub_ids


def concat_segments(
    order: tuple[str, ...],
    segments: dict[str, torch.Tensor],
) -> torch.Tensor:
    return torch.cat([segments[name] for name in order], dim=1)


def coerce_sparse_selection(
    selection: SparseTokenSelection,
    *,
    experiment_mode: str,
) -> SparseTokenSelection | SoftSparseTokenSelection:
    if experiment_mode == "soft_exchange":
        return SoftSparseTokenSelection(
            main_token_indices=selection.main_token_indices,
            ref_token_indices=selection.ref_token_indices,
            token_mask=selection.token_mask,
            token_coords=selection.token_coords,
            crop_token_indices=selection.crop_token_indices,
            crop_token_coords=selection.crop_token_coords,
            crop_bounds=selection.crop_bounds,
        )
    return selection


def resolve_transformer_dir(path: str | Path) -> Path:
    resolved = resolve_path(path)
    if (resolved / "final_transformer").exists():
        return resolved / "final_transformer"
    if (resolved / "transformer").exists() and resolved.name.startswith("checkpoint-"):
        return resolved / "transformer"
    if (resolved / "config.json").exists():
        return resolved
    raise FileNotFoundError(
        f"Could not resolve a transformer directory from {resolved}. "
        "Expected `final_transformer/`, `checkpoint-*/transformer/`, or a diffusers model dir."
    )


def read_train_args(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    resolved = resolve_path(path)
    search_roots = [resolved, resolved.parent, resolved.parent.parent]
    for root in search_roots:
        candidate = root / "train_args.json"
        if candidate.exists():
            return json.loads(candidate.read_text())
    return {}


class SubjectDrivenFlux2Pipeline(Flux2KleinPipeline):
    @torch.no_grad()
    def __call__(
        self,
        *,
        prompt: str | list[str],
        main_image: str | Path | Image.Image,
        subject_image: str | Path | Image.Image | None = None,
        mask: str | Path | Image.Image | torch.Tensor | None = None,
        negative_prompt: str | list[str] | None = None,
        experiment_mode: str = "hard_exchange",
        sub_region_mode: str = "mask",
        height: int = 1024,
        width: int = 1024,
        subject_size: int = 1024,
        drop_subject: bool = False,
        zero_cond_t: bool = False,
        use_sparse_sub_branch: bool = True,
        pe_exchange_region: str = "auto",
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 4.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int, ...] = (9, 18, 27),
        hidden_state_order: tuple[str, ...] | list[str] | str = ("target", "sub", "cond"),
        img_ids_order: tuple[str, ...] | list[str] | str = ("target", "sub", "cond"),
        sub_t_coord: int = 20,
        condition_t_start: int = 40,
        condition_t_stride: int = 20,
        pe_main_t_coord: int = 0,
    ):
        if experiment_mode not in VALID_EXPERIMENT_MODES:
            raise ValueError(f"Unsupported experiment_mode {experiment_mode}.")
        if experiment_mode in {"soft_exchange", "hard_exchange"} and not hasattr(self.transformer, "pe_exchange"):
            raise ValueError(
                f"experiment_mode={experiment_mode} requires a transformer with `pe_exchange`, "
                f"got {self.transformer.__class__.__name__}."
            )

        has_sub = experiment_mode != "no_sub"
        hidden_state_order = filter_order_for_mode(normalize_order(hidden_state_order), has_sub)
        img_ids_order = filter_order_for_mode(normalize_order(img_ids_order), has_sub)

        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=None,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            guidance_scale=guidance_scale,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        device = self._execution_device
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        negative_prompt_embeds = None
        negative_text_ids = None
        if self.do_classifier_free_guidance:
            negative_prompt = "" if negative_prompt is None else negative_prompt
            if isinstance(prompt, list) and isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * len(prompt)
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=None,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        main_pixels = image_to_tensor(main_image, width=width, height=height).to(device=device, dtype=self.vae.dtype)
        main_latents_single = self._encode_vae_image(main_pixels, generator=generator)
        packed_main_single = self._pack_latents(main_latents_single)
        mask_tensor = None
        needs_subject_ref = has_sub and not drop_subject
        if needs_subject_ref and subject_image is None:
            raise ValueError("subject_image is required unless experiment_mode=no_sub or drop_subject=True.")
        if has_sub and mask is None:
            raise ValueError("mask is required unless experiment_mode=no_sub.")

        if has_sub:
            mask_tensor = mask_to_tensor(mask, width=width, height=height).to(device=device)

        if needs_subject_ref:
            # Subject reference is ALWAYS encoded at its native SQUARE training resolution
            # (subject_size x subject_size). The native-aspect cache stores the subject at a
            # fixed 1024x1024 regardless of the target's aspect ratio, so the subject branch
            # must NOT be resized to the (possibly non-square) target width/height. The
            # condition ids are derived per-branch from each latent's own grid, so main
            # (native) and subject (square) can differ in size.
            subject_pixels = image_to_tensor(subject_image, width=subject_size, height=subject_size).to(
                device=device, dtype=self.vae.dtype
            )
            subject_latents_single = self._encode_vae_image(subject_pixels, generator=generator)
            packed_subject_single = self._pack_latents(subject_latents_single)
            packed_cond_single = torch.cat([packed_main_single, packed_subject_single], dim=1)
            cond_image_ids_single = build_condition_ids(
                [main_latents_single, subject_latents_single],
                start_t=condition_t_start,
                stride=condition_t_stride,
                device=device,
            )
        else:
            packed_cond_single = packed_main_single
            cond_image_ids_single = build_condition_ids(
                [main_latents_single],
                start_t=condition_t_start,
                stride=condition_t_stride,
                device=device,
            )

        packed_cond = packed_cond_single.expand(batch_size, -1, -1)
        cond_image_ids = cond_image_ids_single.expand(batch_size, -1, -1)

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        selection = None
        sub_latents = None
        sub_ids = None
        if has_sub:
            selection, sub_latents, sub_ids = build_sparse_sub_branch(
                main_latents_single,
                mask_tensor,
                sub_region_mode=sub_region_mode,
                sub_t_coord=sub_t_coord,
                use_sparse_sub_branch=use_sparse_sub_branch,
                pe_exchange_region=pe_exchange_region,
                generator=generator,
                batch_size=batch_size,
            )
            selection = coerce_sparse_selection(selection, experiment_mode=experiment_mode)

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None

        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        sub_scheduler = copy.deepcopy(self.scheduler) if has_sub else None
        self.scheduler.set_begin_index(0)
        if sub_scheduler is not None:
            sub_scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for index, timestep_value in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = timestep_value
                timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)

                hidden_segments: dict[str, torch.Tensor] = {
                    "target": latents.to(self.transformer.dtype),
                    "cond": packed_cond.to(self.transformer.dtype),
                }
                id_segments: dict[str, torch.Tensor] = {
                    "target": latent_ids,
                    "cond": cond_image_ids,
                }

                if has_sub and sub_latents is not None and sub_ids is not None:
                    hidden_segments["sub"] = sub_latents.to(self.transformer.dtype)
                    id_segments["sub"] = sub_ids

                hidden_states = concat_segments(hidden_state_order, hidden_segments)
                img_ids = concat_segments(img_ids_order, id_segments)

                guidance = None
                if getattr(self.transformer.config, "guidance_embeds", False):
                    guidance = torch.full([batch_size], guidance_scale, device=device, dtype=latents.dtype)

                model_kwargs: dict[str, Any] = {
                    "hidden_states": hidden_states,
                    "timestep": timestep / 1000,
                    "guidance": guidance,
                    "encoder_hidden_states": prompt_embeds,
                    "txt_ids": text_ids,
                    "img_ids": img_ids,
                    "joint_attention_kwargs": self.attention_kwargs,
                    "return_dict": False,
                }
                if experiment_mode in {"soft_exchange", "hard_exchange"} and selection is not None:
                    model_kwargs["pe_sparse_token_selection"] = selection
                    model_kwargs["pe_ref_t_coord"] = sub_t_coord
                    model_kwargs["pe_main_t_coord"] = pe_main_t_coord
                if zero_cond_t and experiment_mode == "hard_exchange":
                    # Match training (--zero_cond_t): the trailing cond tokens
                    # (background[+subject]) are modulated by a SEPARATE t=0 AdaLN path via
                    # `num_cond_tokens`. Without this the PE-exchange transformer defaults to
                    # num_cond_tokens=0 and modulates the background condition at the noisy
                    # denoising timestep -> background reconstruction degrades. Only the
                    # hard-exchange transformer accepts this kwarg; the vanilla base does not.
                    model_kwargs["num_cond_tokens"] = int(packed_cond.shape[1])

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(**model_kwargs)[0]

                if self.do_classifier_free_guidance and negative_prompt_embeds is not None and negative_text_ids is not None:
                    neg_model_kwargs = dict(model_kwargs)
                    neg_model_kwargs["encoder_hidden_states"] = negative_prompt_embeds
                    neg_model_kwargs["txt_ids"] = negative_text_ids
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(**neg_model_kwargs)[0]
                    noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

                segment_slices: dict[str, slice] = {}
                segment_start = 0
                for segment_name in hidden_state_order:
                    segment_end = segment_start + hidden_segments[segment_name].shape[1]
                    segment_slices[segment_name] = slice(segment_start, segment_end)
                    segment_start = segment_end

                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred[:, segment_slices["target"], :],
                    timestep_value,
                    latents,
                    return_dict=False,
                )[0]
                if sub_latents is not None and sub_scheduler is not None:
                    sub_latents = sub_scheduler.step(
                        noise_pred[:, segment_slices["sub"], :],
                        timestep_value,
                        sub_latents,
                        return_dict=False,
                    )[0]

                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for tensor_name in callback_on_step_end_tensor_inputs:
                        callback_kwargs[tensor_name] = locals()[tensor_name]
                    callback_outputs = callback_on_step_end(self, index, timestep_value, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if index == len(timesteps) - 1 or ((index + 1) > num_warmup_steps and (index + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        latent_height = 2 * (int(height) // (self.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self.vae_scale_factor * 2))
        latents = self._unpack_latents_with_ids(latents, latent_ids, latent_height // 2, latent_width // 2)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)
        if output_type == "latent":
            image = latents
        else:
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return Flux2PipelineOutput(images=image)


def build_subject_driven_pipeline(
    *,
    pretrained_model_name_or_path: str = DEFAULT_MODEL,
    experiment_mode: str = "hard_exchange",
    transformer_path: str | Path | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool = True,
    pe_exchange_kwargs: Optional[dict[str, Any]] = None,
    device: str | torch.device | None = None,
    **from_pretrained_kwargs,
) -> SubjectDrivenFlux2Pipeline:
    if experiment_mode not in VALID_EXPERIMENT_MODES:
        raise ValueError(f"Unsupported experiment_mode {experiment_mode}.")

    train_args = read_train_args(transformer_path)
    if train_args.get("pretrained_model_name_or_path") and pretrained_model_name_or_path == DEFAULT_MODEL:
        pretrained_model_name_or_path = train_args["pretrained_model_name_or_path"]

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    direct_kwargs = dict(from_pretrained_kwargs)
    if resolved_device.type == "cuda":
        direct_kwargs.setdefault("device_map", {"": _device_string(resolved_device)})
        direct_kwargs.setdefault("low_cpu_mem_usage", True)

    try:
        base_pipe = Flux2KleinPipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            **direct_kwargs,
        )
    except (TypeError, ValueError, RuntimeError):
        base_pipe = Flux2KleinPipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            **from_pretrained_kwargs,
        )

    transformer = base_pipe.transformer
    if experiment_mode in {"hard_exchange", "soft_exchange"}:
        config = transformer.config
        config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        valid_keys = set(inspect.signature(Flux2Transformer2DModel.__init__).parameters.keys()) - {"self"}
        init_kwargs = {key: value for key, value in config_dict.items() if key in valid_keys}
        merged_exchange_kwargs = {
            "dim_in": transformer.inner_dim,
            "pe_dim": sum(transformer.config.axes_dims_rope),
            "num_layers": len(transformer.transformer_blocks) + len(transformer.single_transformer_blocks),
            "sampler": "vanilla_ste",
            "encoder_layers": int(train_args.get("ste_encoder_layers", 1)),
            "encoder_num_heads": int(train_args.get("ste_encoder_num_heads", 8)),
            "head_init": str(train_args.get("ste_head_init", "zero")),
        }
        if pe_exchange_kwargs:
            merged_exchange_kwargs.update(pe_exchange_kwargs)
        custom_cls = (
            Flux2KleinPESoftExchangeTransformer2DModel
            if experiment_mode == "soft_exchange"
            else Flux2KleinPEExchangeTransformer2DModel
        )
        custom_transformer = custom_cls(pe_exchange_kwargs=merged_exchange_kwargs, **init_kwargs)
        custom_transformer.load_state_dict(transformer.state_dict(), strict=False)
        if transformer_path is not None:
            weights_dir = resolve_transformer_dir(transformer_path)
            loaded_transformer = custom_cls.from_pretrained(
                weights_dir,
                torch_dtype=torch_dtype,
                local_files_only=local_files_only,
            )
            custom_transformer.load_state_dict(loaded_transformer.state_dict(), strict=False)
            del loaded_transformer
        transformer = custom_transformer.to(device=resolved_device, dtype=base_pipe.transformer.dtype)
    elif transformer_path is not None:
        weights_dir = resolve_transformer_dir(transformer_path)
        transformer = Flux2Transformer2DModel.from_pretrained(
            weights_dir,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
        transformer = transformer.to(device=resolved_device, dtype=torch_dtype)

    pipe = SubjectDrivenFlux2Pipeline(
        scheduler=base_pipe.scheduler,
        vae=base_pipe.vae,
        text_encoder=base_pipe.text_encoder,
        tokenizer=base_pipe.tokenizer,
        transformer=transformer,
        is_distilled=base_pipe.config.is_distilled,
    )
    return _materialize_pipeline_modules(pipe, device=resolved_device, dtype=torch_dtype)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Subject-driven FLUX2 Klein generation pipeline.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--main-image", type=Path, required=True)
    parser.add_argument("--subject-image", type=Path, default=None)
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--drop-subject", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument(
        "--experiment-mode",
        choices=tuple(sorted(VALID_EXPERIMENT_MODES)),
        default="hard_exchange",
    )
    parser.add_argument(
        "--layout",
        choices=("auto", "t60", "ids20"),
        default="auto",
        help=(
            "Timestep-axis preset only. "
            "`t60` => sub_t=60 with cond_t starting at 20; "
            "`ids20` => sub_t=20 with cond_t starting at 40. "
            "Token order and img_ids order are controlled separately by "
            "`--hidden-order` and `--img-ids-order`."
        ),
    )
    parser.add_argument("--hidden-order", default=None, help="Comma-separated permutation of target,cond,sub.")
    parser.add_argument("--img-ids-order", default=None, help="Comma-separated permutation of target,cond,sub.")
    parser.add_argument("--sub-t-coord", type=int, default=None)
    parser.add_argument("--condition-t-start", type=int, default=None)
    parser.add_argument("--condition-t-stride", type=int, default=20)
    parser.add_argument("--sub-region-mode", choices=("mask", "bbox"), default="mask")
    parser.add_argument(
        "--sub-branch-layout",
        choices=("sparse_mask", "dense_bbox"),
        default="sparse_mask",
    )
    parser.add_argument(
        "--pe-exchange-region",
        choices=("auto", "mask", "bbox"),
        default="auto",
    )
    parser.add_argument("--pretrained-model-name-or-path", default=DEFAULT_MODEL)
    parser.add_argument("--transformer-path", type=Path, default=None)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text-encoder-out-layers", default="9,18,27")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    layout = resolve_layout(
        experiment_mode=args.experiment_mode,
        layout=args.layout,
        hidden_state_order=args.hidden_order,
        img_ids_order=args.img_ids_order,
        sub_t_coord=args.sub_t_coord,
        condition_t_start=args.condition_t_start,
        condition_t_stride=args.condition_t_stride,
    )
    dtype = to_dtype(args.dtype)
    device = torch.device(args.device)

    pipe = build_subject_driven_pipeline(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        experiment_mode=args.experiment_mode,
        transformer_path=args.transformer_path,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        main_image=args.main_image,
        subject_image=args.subject_image,
        mask=args.mask,
        experiment_mode=args.experiment_mode,
        sub_region_mode=args.sub_region_mode,
        use_sparse_sub_branch=args.sub_branch_layout == "sparse_mask",
        pe_exchange_region=args.pe_exchange_region,
        height=args.height,
        width=args.width,
        drop_subject=args.drop_subject,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        text_encoder_out_layers=parse_text_layers(args.text_encoder_out_layers),
        hidden_state_order=layout.hidden_state_order,
        img_ids_order=layout.img_ids_order,
        sub_t_coord=layout.sub_t_coord,
        condition_t_start=layout.condition_t_start,
        condition_t_stride=layout.condition_t_stride,
    )

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "experiment_mode": args.experiment_mode,
                "timestep_layout": layout.timestep_layout,
                "hidden_order": list(layout.hidden_state_order),
                "img_ids_order": list(layout.img_ids_order),
                "sub_t_coord": layout.sub_t_coord,
                "condition_t_start": layout.condition_t_start,
                "condition_t_stride": layout.condition_t_stride,
                "height": args.height,
                "width": args.width,
                "num_inference_steps": args.num_inference_steps,
                "seed": args.seed,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
