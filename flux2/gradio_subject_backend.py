from __future__ import annotations

import copy
import glob
import inspect
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parent
LOCAL_DIFFUSERS_SRC = REPO_ROOT / "diffusers" / "src"
if LOCAL_DIFFUSERS_SRC.exists() and str(LOCAL_DIFFUSERS_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIFFUSERS_SRC))

from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel  # noqa: E402
from diffusers.guiders.classifier_free_zero_star_guidance import cfg_zero_star_scale  # noqa: E402
from diffusers.pipelines.flux2.pipeline_flux2_klein import (  # noqa: E402
    compute_empirical_mu,
    retrieve_timesteps,
)
from flux2_klein_pe_exchange import Flux2KleinPEExchangeTransformer2DModel  # noqa: E402
from subject_driven_generation_pipeline import (  # noqa: E402
    SubjectDrivenFlux2Pipeline,
    build_condition_ids,
    build_sparse_sub_branch,
    coerce_sparse_selection,
    image_to_tensor,
    mask_to_tensor,
)


DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-base-9B"
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "flux2_klein_hard_exchange_8gpu_bs2_gc_eager"
DEFAULT_DTYPE = torch.bfloat16
DEFAULT_SUBJECT_SIZE = 1024
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 4.0
DEFAULT_SUB_T = 20
DEFAULT_COND_T_START = 40
DEFAULT_COND_T_STRIDE = 20
DEFAULT_TEXT_ENCODER_LAYERS = (9, 18, 27)
DEFAULT_EXPERIMENT_MODE = "hard_exchange"
DEFAULT_STE_MASK_OUTPUT_DIR = REPO_ROOT / "gradio_outputs" / "ste_masks"
MAX_SUBJECT_REFERENCE_IMAGES = 3
VALID_SUB_ID_SPATIAL_MODES = ("global", "local")


def rescale_cfg_prediction(
    guided_prediction: torch.Tensor,
    conditional_prediction: torch.Tensor,
    guidance_rescale: float,
) -> torch.Tensor:
    """Apply CFG Rescale from Lin et al. (2024) with a safe zero-variance guard."""
    strength = float(guidance_rescale)
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError(f"guidance_rescale must be finite and within [0, 1], got {strength}.")
    if strength == 0.0:
        return guided_prediction
    if guided_prediction.shape != conditional_prediction.shape:
        raise ValueError(
            "CFG and conditional predictions must have identical shapes, got "
            f"{tuple(guided_prediction.shape)} and {tuple(conditional_prediction.shape)}."
        )

    reduce_dims = tuple(range(1, guided_prediction.ndim))
    conditional_std = conditional_prediction.float().std(dim=reduce_dims, keepdim=True)
    guided_std = guided_prediction.float().std(dim=reduce_dims, keepdim=True)
    safe_guided_std = guided_std.clamp_min(torch.finfo(torch.float32).eps)
    rescale_factor = conditional_std / safe_guided_std
    blend_factor = strength * rescale_factor + (1.0 - strength)
    return guided_prediction * blend_factor.to(guided_prediction.dtype)


def apply_cfg_zero_star_guidance(
    conditional_prediction: torch.Tensor,
    unconditional_prediction: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Apply Diffusers' CFG-Zero* optimized unconditional scale to one ODE state."""
    if conditional_prediction.shape != unconditional_prediction.shape:
        raise ValueError(
            "Conditional and unconditional predictions must have identical shapes, got "
            f"{tuple(conditional_prediction.shape)} and {tuple(unconditional_prediction.shape)}."
        )
    conditional_flat = conditional_prediction.flatten(1)
    unconditional_flat = unconditional_prediction.flatten(1)
    alpha = cfg_zero_star_scale(conditional_flat, unconditional_flat)
    alpha = alpha.view(-1, *(1,) * (conditional_prediction.ndim - 1))
    scaled_unconditional = unconditional_prediction * alpha
    return scaled_unconditional + float(guidance_scale) * (
        conditional_prediction - scaled_unconditional
    )


@dataclass(frozen=True)
class GenerationResult:
    main_image: Image.Image
    sub_image: Image.Image | None
    metadata: dict[str, Any]


def _area_normalized_size(
    width: int,
    height: int,
    *,
    resolution: int = DEFAULT_SUBJECT_SIZE,
    multiple_of: int = 16,
) -> tuple[int, int]:
    """Preserve aspect ratio while normalizing area to approximately resolution squared."""
    width = int(width)
    height = int(height)
    resolution = int(resolution)
    multiple_of = int(multiple_of)
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {(width, height)}.")
    if resolution <= 0 or multiple_of <= 0:
        raise ValueError(
            f"resolution and multiple_of must be positive, got {(resolution, multiple_of)}."
        )

    scale = math.sqrt((resolution * resolution) / (width * height))
    snapped_w = max(multiple_of, round(width * scale / multiple_of) * multiple_of)
    snapped_h = max(multiple_of, round(height * scale / multiple_of) * multiple_of)
    return snapped_w, snapped_h


def _generator_for(device: torch.device, seed: int) -> torch.Generator:
    if device.type == "cuda":
        dev = f"cuda:{device.index}" if device.index is not None else "cuda"
        return torch.Generator(device=dev).manual_seed(int(seed))
    return torch.Generator().manual_seed(int(seed))


def _device_string(device: torch.device) -> str:
    if device.type == "cuda" and device.index is not None:
        return f"cuda:{device.index}"
    return device.type


def _combine_layer_gifs(
    gif_paths: list[Path],
    output_path: Path,
    *,
    grid_columns: int,
    duration_ms: int,
) -> int:
    """Combine synchronized per-layer GIFs into one layer-grid GIF."""
    if not gif_paths:
        raise ValueError("At least one layer GIF is required.")
    if grid_columns <= 0:
        raise ValueError("grid_columns must be positive.")

    layer_gifs = [Image.open(path) for path in gif_paths]
    try:
        frame_counts = {int(gif.n_frames) for gif in layer_gifs}
        frame_sizes = {gif.size for gif in layer_gifs}
        if len(frame_counts) != 1:
            raise RuntimeError(f"Layer GIF frame counts differ: {sorted(frame_counts)}")
        if len(frame_sizes) != 1:
            raise RuntimeError(f"Layer GIF frame sizes differ: {sorted(frame_sizes)}")

        frame_count = frame_counts.pop()
        tile_w, tile_h = frame_sizes.pop()
        grid_rows = math.ceil(len(layer_gifs) / grid_columns)
        combined_frames: list[Image.Image] = []
        for frame_index in range(frame_count):
            canvas = Image.new("L", (grid_columns * tile_w, grid_rows * tile_h), 0)
            for position, layer_gif in enumerate(layer_gifs):
                layer_gif.seek(frame_index)
                row, column = divmod(position, grid_columns)
                canvas.paste(layer_gif.convert("L"), (column * tile_w, row * tile_h))
            combined_frames.append(canvas)

        combined_frames[0].save(
            output_path,
            save_all=True,
            append_images=combined_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
        return frame_count
    finally:
        for layer_gif in layer_gifs:
            layer_gif.close()


class SubjectDrivenGradioBackend:
    def __init__(
        self,
        *,
        run_dir: str | Path = DEFAULT_RUN_DIR,
        checkpoint_path: str | Path | None = None,
        pretrained_model_name_or_path: str = DEFAULT_MODEL,
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype = DEFAULT_DTYPE,
        local_files_only: bool = False,
        subject_size: int = DEFAULT_SUBJECT_SIZE,
        num_inference_steps: int = DEFAULT_STEPS,
        default_guidance_scale: float = DEFAULT_GUIDANCE,
        text_encoder_out_layers: tuple[int, ...] = DEFAULT_TEXT_ENCODER_LAYERS,
        default_experiment_mode: str = DEFAULT_EXPERIMENT_MODE,
        ste_mask_output_dir: str | Path | None = None,
        sub_region_mode: str = "mask",
        use_sparse_sub_branch: bool = True,
        pe_exchange_region: str = "mask",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.torch_dtype = torch_dtype
        self.local_files_only = local_files_only
        self.subject_size = int(subject_size)
        self.num_inference_steps = int(num_inference_steps)
        self.default_guidance_scale = float(default_guidance_scale)
        self.text_encoder_out_layers = tuple(int(x) for x in text_encoder_out_layers)
        self.default_experiment_mode = str(default_experiment_mode)
        self.ste_mask_output_dir = Path(ste_mask_output_dir or DEFAULT_STE_MASK_OUTPUT_DIR)
        if sub_region_mode not in {"mask", "bbox"}:
            raise ValueError(f"Unsupported sub_region_mode: {sub_region_mode!r}.")
        if pe_exchange_region not in {"mask", "bbox"}:
            raise ValueError(f"Unsupported pe_exchange_region: {pe_exchange_region!r}.")
        if use_sparse_sub_branch and pe_exchange_region == "bbox":
            raise ValueError("bbox-wide PE exchange requires the dense sub branch.")
        self.sub_region_mode = sub_region_mode
        self.use_sparse_sub_branch = bool(use_sparse_sub_branch)
        self.pe_exchange_region = pe_exchange_region

        self.checkpoint_dir, self.checkpoint_format = self._resolve_checkpoint()
        self.pipe, self.load_metadata = self._build_pipeline()
        self.last_cross_attention_collection: dict[str, Any] | None = None
        self._latents_bn_mean = self.pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(
            self.device, self.torch_dtype
        )
        self._latents_bn_std = torch.sqrt(
            self.pipe.vae.bn.running_var.view(1, -1, 1, 1) + self.pipe.vae.config.batch_norm_eps
        ).to(self.device, self.torch_dtype)

    def _resolve_checkpoint(self) -> tuple[Path, str]:
        if self.checkpoint_path is not None:
            return self._checkpoint_format(self.checkpoint_path)

        candidates: list[tuple[int, Path, str]] = []
        for raw in glob.glob(str(self.run_dir / "checkpoint-*")):
            match = re.search(r"checkpoint-(\d+)$", raw)
            if match is None:
                continue
            step = int(match.group(1))
            path, fmt = self._checkpoint_format(Path(raw))
            candidates.append((step, path, fmt))

        if not candidates:
            raise FileNotFoundError(f"No complete checkpoints found under {self.run_dir}.")

        candidates.sort(key=lambda x: x[0])
        _, checkpoint_dir, checkpoint_format = candidates[-1]
        return checkpoint_dir, checkpoint_format

    def _checkpoint_format(self, path: str | Path) -> tuple[Path, str]:
        path = Path(path)
        if path.name == "transformer" and path.is_dir():
            return path.parent, "hf"
        transformer_dir = path / "transformer"
        if transformer_dir.is_dir() and list(transformer_dir.glob("*.safetensors")):
            return path, "hf"
        zero_file = path / "pytorch_model" / "mp_rank_00_model_states.pt"
        if zero_file.exists():
            return path, "zero"
        raise FileNotFoundError(
            f"Could not resolve checkpoint format from {path}. "
            "Expected checkpoint-*/transformer/*.safetensors or "
            "checkpoint-*/pytorch_model/mp_rank_00_model_states.pt."
        )

    def _build_pe_transformer(
        self,
    ) -> tuple[Flux2KleinPEExchangeTransformer2DModel, list[str], list[str]]:
        if self.checkpoint_format != "hf":
            raise ValueError(
                "The no-base-transformer loader requires a consolidated HF checkpoint under "
                "checkpoint-*/transformer."
            )

        transformer_dir = self.checkpoint_dir / "transformer"
        config_path = transformer_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing trained transformer config: {config_path}")
        config_dict = json.loads(config_path.read_text())
        valid_keys = set(inspect.signature(Flux2Transformer2DModel.__init__).parameters.keys()) - {"self"}
        init_kwargs = {k: v for k, v in config_dict.items() if k in valid_keys}
        inner_dim = int(init_kwargs["num_attention_heads"]) * int(init_kwargs["attention_head_dim"])
        total_layers = int(init_kwargs["num_layers"]) + int(init_kwargs["num_single_layers"])
        pe_kwargs = {
            "dim_in": inner_dim,
            "pe_dim": sum(int(value) for value in init_kwargs["axes_dims_rope"]),
            "num_layers": total_layers,
            "sampler": "vanilla_ste",
            "encoder_layers": 1,
            "encoder_num_heads": 8,
            "head_init": "zero",
        }
        from accelerate import init_empty_weights
        from accelerate.utils.modeling import load_checkpoint_in_model

        with init_empty_weights():
            model = Flux2KleinPEExchangeTransformer2DModel(
                pe_exchange_kwargs=pe_kwargs,
                **init_kwargs,
            )

        index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing trained transformer shard index: {index_path}")
        checkpoint_keys = set(json.loads(index_path.read_text())["weight_map"])
        model_keys = set(model.state_dict())
        parameter_keys = set(dict(model.named_parameters()))
        missing_parameters = sorted(parameter_keys - checkpoint_keys)
        if missing_parameters:
            raise RuntimeError(
                f"Trained transformer checkpoint is missing {len(missing_parameters)} parameters, "
                f"for example {missing_parameters[:5]}."
            )
        missing = sorted(model_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - model_keys)
        if unexpected:
            raise RuntimeError(
                f"Trained transformer checkpoint has {len(unexpected)} unexpected keys, "
                f"for example {unexpected[:5]}."
            )

        load_checkpoint_in_model(
            model,
            checkpoint=transformer_dir,
            device_map={"": _device_string(self.device)},
            dtype=self.torch_dtype,
            strict=False,
        )
        model.eval()
        return model, missing, unexpected

    def _direct_load_kwargs(self) -> dict[str, Any]:
        if self.device.type != "cuda":
            return {}
        # Diffusers accepts the direct-placement strategy "cuda", not an
        # indexed device-map string such as "cuda:0".  CUDA_VISIBLE_DEVICES
        # makes the selected physical GPU logical cuda:0 for the launchers.
        # Keep the CPU-load fallback for an unmasked, explicitly indexed GPU.
        if self.device.index not in (None, 0):
            return {}
        return {
            "device_map": "cuda",
            "low_cpu_mem_usage": True,
        }

    def _build_pipeline(self) -> tuple[SubjectDrivenFlux2Pipeline, dict[str, Any]]:
        direct_gpu_init = False
        direct_gpu_init_error = None
        base_kwargs = {
            "torch_dtype": self.torch_dtype,
            "local_files_only": self.local_files_only,
            "transformer": None,
        }
        direct_load_kwargs = self._direct_load_kwargs()
        try:
            base = Flux2KleinPipeline.from_pretrained(
                self.pretrained_model_name_or_path,
                **base_kwargs,
                **direct_load_kwargs,
            )
            direct_gpu_init = bool(direct_load_kwargs)
        except (TypeError, ValueError, RuntimeError) as exc:
            if not direct_load_kwargs:
                raise
            direct_gpu_init_error = str(exc)
            base = Flux2KleinPipeline.from_pretrained(
                self.pretrained_model_name_or_path,
                **base_kwargs,
            )
        transformer, missing, unexpected = self._build_pe_transformer()
        scheduler = base.scheduler
        vae = base.vae
        text_encoder = base.text_encoder
        tokenizer = base.tokenizer
        is_distilled = base.config.is_distilled
        del base

        pipe = SubjectDrivenFlux2Pipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            is_distilled=is_distilled,
        )
        pipe.transformer = pipe.transformer.to(self.device, dtype=self.torch_dtype)
        pipe.vae = pipe.vae.to(self.device, dtype=self.torch_dtype)
        pipe.text_encoder = pipe.text_encoder.to(self.device, dtype=self.torch_dtype)

        metadata = {
            "checkpoint_dir": str(self.checkpoint_dir),
            "checkpoint_format": self.checkpoint_format,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "device": str(self.device),
            "dtype": str(self.torch_dtype),
            "direct_gpu_init": direct_gpu_init,
            "direct_gpu_init_error": direct_gpu_init_error,
            "base_transformer_loaded": False,
        }
        return pipe, metadata

    def _decode_grid(self, grid_dchw: torch.Tensor) -> Image.Image:
        grid = grid_dchw.to(self.device, self.torch_dtype)
        grid = grid * self._latents_bn_std + self._latents_bn_mean
        image = self.pipe.vae.decode(self.pipe._unpatchify_latents(grid), return_dict=False)[0]
        return self.pipe.image_processor.postprocess(image, output_type="pil")[0]

    def _unpack_target(self, packed: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        token_h = int(ids[..., 1].max().item()) + 1
        token_w = int(ids[..., 2].max().item()) + 1
        return self.pipe._unpack_latents_with_ids(packed.to(self.torch_dtype), ids, token_h, token_w)

    def _save_ste_masks(
        self,
        gate_collection: dict[str, Any] | None,
        *,
        width: int,
        height: int,
        denoising_schedule: list[dict[str, float | int | None]] | None = None,
        sub_id_spatial_mode: str = "global",
    ) -> dict[str, Any] | None:
        if gate_collection is None:
            return None

        raw_gates_by_layer = gate_collection["raw_gates_by_layer"]
        coords = gate_collection["main_token_coords"].to(torch.long)
        if not raw_gates_by_layer or coords.numel() == 0:
            return None

        token_h = height // 16
        token_w = width // 16
        if int(coords[:, 0].max().item()) >= token_h or int(coords[:, 1].max().item()) >= token_w:
            raise RuntimeError("STE gate coordinates exceed the generated main-image token grid.")
        def make_token_grid(values: torch.Tensor, *, binary: bool) -> torch.Tensor:
            grid = torch.zeros((token_h, token_w), dtype=torch.uint8)
            if binary:
                values = values.to(torch.uint8) * 255
            grid[coords[:, 0], coords[:, 1]] = values.to(torch.uint8)
            return grid

        layer_records: list[dict[str, Any]] = []
        for layer_idx, raw_gates in sorted(raw_gates_by_layer.items()):
            raw_gates = raw_gates.to(torch.float32)
            if raw_gates.ndim != 2 or raw_gates.shape[1] != coords.shape[0]:
                raise RuntimeError(
                    "Each STE layer must have gate shape [timesteps, selected_tokens] matching the coordinates."
                )
            if denoising_schedule is not None and raw_gates.shape[0] != len(denoising_schedule):
                raise RuntimeError(
                    "Each STE layer must have one conditional gate record per denoising timestep."
                )
            keep_rate = raw_gates.mean(dim=0)
            keep_majority = keep_rate >= 0.5
            exchange_majority = ~keep_majority
            layer_records.append(
                {
                    "layer_idx": int(layer_idx),
                    "timestep_records": int(raw_gates.shape[0]),
                    "hard_gates": raw_gates.to(torch.uint8),
                    "keep_tokens": make_token_grid(keep_majority, binary=True),
                    "exchange_tokens": make_token_grid(exchange_majority, binary=True),
                    "exchange_rate_tokens": make_token_grid(
                        (1.0 - keep_rate).mul(255).round().to(torch.uint8),
                        binary=False,
                    ),
                    "keep_fraction": float(keep_majority.to(torch.float32).mean().item()),
                    "exchange_fraction": float(exchange_majority.to(torch.float32).mean().item()),
                }
            )

        layer_count = len(layer_records)
        grid_columns = math.ceil(math.sqrt(layer_count))
        grid_rows = math.ceil(layer_count / grid_columns)

        def make_layer_canvas(key: str) -> Image.Image:
            canvas = torch.zeros((grid_rows * token_h, grid_columns * token_w), dtype=torch.uint8)
            for position, layer_record in enumerate(layer_records):
                row, column = divmod(position, grid_columns)
                y0 = row * token_h
                x0 = column * token_w
                canvas[y0 : y0 + token_h, x0 : x0 + token_w] = layer_record[key]
            return Image.fromarray(canvas.numpy(), mode="L")

        keep_grid_tokens = make_layer_canvas("keep_tokens")
        exchange_grid_tokens = make_layer_canvas("exchange_tokens")
        exchange_rate_grid_tokens = make_layer_canvas("exchange_rate_tokens")
        display_scale = max(1, 128 // min(token_h, token_w))

        generation_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid4().hex[:8]}"
        output_dir = self.ste_mask_output_dir / generation_id
        output_dir.mkdir(parents=True, exist_ok=False)

        keep_grid_token_path = output_dir / "ste_layer_keep_grid_tokens.png"
        exchange_grid_token_path = output_dir / "ste_layer_exchange_grid_tokens.png"
        exchange_rate_grid_token_path = output_dir / "ste_layer_exchange_rate_grid_tokens.png"
        keep_grid_path = output_dir / "ste_layer_keep_grid.png"
        exchange_grid_path = output_dir / "ste_layer_exchange_grid.png"
        exchange_rate_grid_path = output_dir / "ste_layer_exchange_rate_grid.png"
        layout_path = output_dir / "ste_layer_layout.json"
        stats_path = output_dir / "ste_mask_stats.json"
        raw_trace_path = output_dir / "ste_raw_trace.safetensors"
        raw_summary_path = output_dir / "ste_raw_summary.json"
        timestep_root = output_dir / "ste_layer_exchange_by_timestep"
        timestep_manifest_path = output_dir / "ste_timestep_manifest.json"
        timestep_grid_gif_path = output_dir / "ste_layer_exchange_timestep_grid.gif"

        keep_grid_tokens.save(keep_grid_token_path)
        exchange_grid_tokens.save(exchange_grid_token_path)
        exchange_rate_grid_tokens.save(exchange_rate_grid_token_path)
        display_size = (exchange_grid_tokens.width * display_scale, exchange_grid_tokens.height * display_scale)
        keep_grid_tokens.resize(display_size, Image.Resampling.NEAREST).save(keep_grid_path)
        exchange_grid_tokens.resize(display_size, Image.Resampling.NEAREST).save(exchange_grid_path)
        exchange_rate_grid_tokens.resize(display_size, Image.Resampling.NEAREST).save(exchange_rate_grid_path)

        timestep_root.mkdir()
        gif_duration_ms = 200
        gif_scale = max(1, min(8, 256 // min(token_h, token_w)))
        timestep_layers: list[dict[str, Any]] = []
        for layer_record in layer_records:
            layer_idx = int(layer_record["layer_idx"])
            hard_gates = layer_record["hard_gates"]
            layer_dir = timestep_root / f"layer_{layer_idx:02d}"
            layer_dir.mkdir()
            gif_frames: list[Image.Image] = []
            frame_paths: list[str] = []
            for step_index, hard_gate in enumerate(hard_gates):
                exchange_tokens = make_token_grid(1 - hard_gate, binary=True)
                token_image = Image.fromarray(exchange_tokens.numpy(), mode="L")
                frame_path = layer_dir / f"step_{step_index:02d}.png"
                token_image.save(frame_path)
                frame_paths.append(str(frame_path))

                display_frame = token_image.resize(
                    (token_w * gif_scale, token_h * gif_scale),
                    Image.Resampling.NEAREST,
                )
                label = f"Layer {layer_idx:02d} | step {step_index + 1:02d}/{len(hard_gates):02d}"
                if denoising_schedule is not None:
                    label += f" | t={float(denoising_schedule[step_index]['timestep']):.1f}"
                gif_frame = Image.new("L", (max(280, display_frame.width), display_frame.height + 20), 0)
                ImageDraw.Draw(gif_frame).text((4, 4), label, fill=255)
                gif_frame.paste(display_frame, (0, 20))
                gif_frames.append(gif_frame)

            gif_path = timestep_root / f"layer_{layer_idx:02d}_exchange.gif"
            gif_frames[0].save(
                gif_path,
                save_all=True,
                append_images=gif_frames[1:],
                duration=gif_duration_ms,
                loop=0,
                optimize=False,
                disposal=2,
            )
            timestep_layers.append(
                {
                    "layer_idx": layer_idx,
                    "frame_count": len(gif_frames),
                    "frame_directory": str(layer_dir),
                    "frames": frame_paths,
                    "gif": str(gif_path),
                }
            )

        timestep_grid_frame_count = _combine_layer_gifs(
            [Path(layer["gif"]) for layer in timestep_layers],
            timestep_grid_gif_path,
            grid_columns=grid_columns,
            duration_ms=gif_duration_ms,
        )
        timestep_manifest = {
            "schema_version": 1,
            "frame_semantics": {"0": "keep original sub PE", "255": "use swapped main-image PE"},
            "branch": "cond",
            "gif_duration_ms": gif_duration_ms,
            "gif_loop": 0,
            "grid_gif": str(timestep_grid_gif_path),
            "grid_gif_frame_count": timestep_grid_frame_count,
            "grid_columns": grid_columns,
            "grid_rows": grid_rows,
            "denoising_schedule": denoising_schedule or [],
            "layers": timestep_layers,
        }
        timestep_manifest_path.write_text(
            json.dumps(timestep_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        layout = {
            "grid_columns": grid_columns,
            "grid_rows": grid_rows,
            "tile_token_size": [token_w, token_h],
            "display_scale": display_scale,
            "layer_order": [
                {
                    "layer_idx": layer_record["layer_idx"],
                    "grid_row": position // grid_columns,
                    "grid_column": position % grid_columns,
                    "timestep_records": layer_record["timestep_records"],
                    "keep_fraction": layer_record["keep_fraction"],
                    "exchange_fraction": layer_record["exchange_fraction"],
                }
                for position, layer_record in enumerate(layer_records)
            ],
        }
        layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")

        raw_trace_saved = False
        raw_summary = None
        if gate_collection.get("capture_raw"):
            raw_tensors: dict[str, torch.Tensor] = {
                "main_token_coords": gate_collection["main_token_coords"].to(torch.int64).contiguous(),
            }
            ref_token_ids = gate_collection.get("ref_token_ids")
            if ref_token_ids is not None:
                raw_tensors["ref_token_ids"] = ref_token_ids.to(torch.int64).contiguous()
            if denoising_schedule:
                raw_tensors["scheduler_timesteps"] = torch.tensor(
                    [float(row["timestep"]) for row in denoising_schedule],
                    dtype=torch.float32,
                )
                if all(row.get("sigma") is not None for row in denoising_schedule):
                    raw_tensors["scheduler_sigmas"] = torch.tensor(
                        [float(row["sigma"]) for row in denoising_schedule],
                        dtype=torch.float32,
                    )

            branches = gate_collection.get("branches") or {}
            raw_summary = {
                "schema_version": 1,
                "hard_gate_semantics": {"0": "exchange", "1": "keep"},
                "sub_id_spatial_mode": sub_id_spatial_mode,
                "denoising_schedule": denoising_schedule or [],
                "branches": {},
                "cond_uncond": {},
            }
            for branch, branch_layers in sorted(branches.items()):
                branch_summary: dict[str, Any] = {}
                for layer_idx, metrics in sorted(branch_layers.items()):
                    for metric, tensor in metrics.items():
                        raw_tensors[f"{branch}.layer_{int(layer_idx):02d}.{metric}"] = tensor.contiguous()

                    hard = metrics["hard_gate"].to(torch.float32)
                    exchange = 1.0 - hard
                    layer_summary: dict[str, Any] = {
                        "records": int(hard.shape[0]),
                        "selected_tokens": int(hard.shape[1]),
                        "exchange_rate": float(exchange.mean().item()),
                        "per_timestep_exchange_rate": [float(x) for x in exchange.mean(dim=1).tolist()],
                    }
                    if "logits" in metrics:
                        logits = metrics["logits"].to(torch.float32)
                        absolute_margin = logits.abs().flatten()
                        layer_summary.update(
                            {
                                "logit_q05": float(torch.quantile(logits.flatten(), 0.05).item()),
                                "logit_q50": float(torch.quantile(logits.flatten(), 0.50).item()),
                                "logit_q95": float(torch.quantile(logits.flatten(), 0.95).item()),
                                "abs_logit_margin_q05": float(torch.quantile(absolute_margin, 0.05).item()),
                                "abs_logit_margin_q50": float(torch.quantile(absolute_margin, 0.50).item()),
                                "abs_logit_margin_q95": float(torch.quantile(absolute_margin, 0.95).item()),
                                "abs_logit_lt_0_01": float((absolute_margin < 0.01).float().mean().item()),
                                "abs_logit_lt_0_1": float((absolute_margin < 0.1).float().mean().item()),
                                "abs_logit_lt_0_5": float((absolute_margin < 0.5).float().mean().item()),
                                "per_timestep_mean_logit": [float(x) for x in logits.mean(dim=1).tolist()],
                            }
                        )
                    if "probs" in metrics:
                        probs = metrics["probs"].to(torch.float32)
                        layer_summary.update(
                            {
                                "prob_q05": float(torch.quantile(probs.flatten(), 0.05).item()),
                                "prob_q50": float(torch.quantile(probs.flatten(), 0.50).item()),
                                "prob_q95": float(torch.quantile(probs.flatten(), 0.95).item()),
                                "per_timestep_mean_keep_probability": [
                                    float(x) for x in probs.mean(dim=1).tolist()
                                ],
                            }
                        )
                    branch_summary[str(layer_idx)] = layer_summary
                raw_summary["branches"][branch] = branch_summary

            cond_layers = branches.get("cond", {})
            uncond_layers = branches.get("uncond", {})
            for layer_idx in sorted(set(cond_layers) & set(uncond_layers)):
                cond_metrics = cond_layers[layer_idx]
                uncond_metrics = uncond_layers[layer_idx]
                cond_hard = cond_metrics["hard_gate"].to(torch.uint8)
                uncond_hard = uncond_metrics["hard_gate"].to(torch.uint8)
                comparison: dict[str, float] = {
                    "hard_gate_disagreement": float((cond_hard != uncond_hard).float().mean().item())
                }
                if "logits" in cond_metrics and "logits" in uncond_metrics:
                    comparison["mean_abs_logit_difference"] = float(
                        (cond_metrics["logits"].float() - uncond_metrics["logits"].float()).abs().mean().item()
                    )
                if "probs" in cond_metrics and "probs" in uncond_metrics:
                    comparison["mean_abs_probability_difference"] = float(
                        (cond_metrics["probs"].float() - uncond_metrics["probs"].float()).abs().mean().item()
                    )
                raw_summary["cond_uncond"][str(layer_idx)] = comparison

            save_file(
                raw_tensors,
                str(raw_trace_path),
                metadata={
                    "schema": "subjectdriven.ste_raw_trace.v1",
                    "sub_id_spatial_mode": sub_id_spatial_mode,
                },
            )
            raw_summary_path.write_text(
                json.dumps(raw_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            raw_trace_saved = True

        stats = {
            "layer_count": layer_count,
            "conditional_timestep_records_per_layer": {
                str(layer_record["layer_idx"]): layer_record["timestep_records"] for layer_record in layer_records
            },
            "token_grid": [token_h, token_w],
            "image_size": [width, height],
            "sub_id_spatial_mode": sub_id_spatial_mode,
            "pe_exchange_region": self.pe_exchange_region,
            "aggregation": "per-layer, per-token conditional majority vote across all denoising timesteps",
            "coverage": (
                "selected sub tokens whose PE participates in exchange, mapped onto the main-image grid; "
                "bbox mode covers the complete dense bbox"
            ),
            "raw_ste_gate": {"0": "use swapped main-image PE for the selected sub token", "1": "keep original sub PE"},
            "exchange_mask": {"0": "keep original sub PE", "1": "use swapped main-image PE"},
            "paths": {
                "layer_keep_grid_tokens": str(keep_grid_token_path),
                "layer_exchange_grid_tokens": str(exchange_grid_token_path),
                "layer_exchange_rate_grid_tokens": str(exchange_rate_grid_token_path),
                "layer_keep_grid": str(keep_grid_path),
                "layer_exchange_grid": str(exchange_grid_path),
                "layer_exchange_rate_grid": str(exchange_rate_grid_path),
                "layer_layout": str(layout_path),
                "layer_exchange_by_timestep": str(timestep_root),
                "layer_exchange_timestep_grid_gif": str(timestep_grid_gif_path),
                "timestep_manifest": str(timestep_manifest_path),
                "stats": str(stats_path),
                "raw_trace": str(raw_trace_path) if raw_trace_saved else None,
                "raw_summary": str(raw_summary_path) if raw_trace_saved else None,
            },
        }
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        return {"directory": str(output_dir), **stats}

    @torch.no_grad()
    def generate(
        self,
        *,
        prompt: str,
        main_image: Image.Image,
        subject_image: Image.Image | Sequence[Image.Image] | None = None,
        mask: Image.Image | None = None,
        seed: int = 0,
        guidance_scale: float | None = None,
        guidance_rescale: float = 0.0,
        use_cfg_zero_star: bool = False,
        cfg_zero_star_zero_init_steps: int = 1,
        use_subject: bool | None = None,
        full_sub_without_mask: bool = False,
        sub_id_spatial_mode: str = "global",
        save_raw_ste: bool = False,
        force_gate_overrides: dict[int, float] | None = None,
        cross_attention_layers: tuple[int, ...] | list[int] | None = None,
        cross_attention_branches: tuple[str, ...] = ("cond",),
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> GenerationResult:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")
        if main_image is None:
            raise ValueError("Main image is required.")
        if sub_id_spatial_mode not in VALID_SUB_ID_SPATIAL_MODES:
            raise ValueError(
                f"sub_id_spatial_mode must be one of {VALID_SUB_ID_SPATIAL_MODES}, got {sub_id_spatial_mode!r}."
            )

        if subject_image is None:
            subject_images: list[Image.Image] = []
        elif isinstance(subject_image, Image.Image):
            subject_images = [subject_image]
        elif isinstance(subject_image, Sequence) and not isinstance(subject_image, (str, bytes)):
            subject_images = list(subject_image)
        else:
            raise TypeError("subject_image must be a PIL image or a sequence of PIL images.")
        if not all(isinstance(image, Image.Image) for image in subject_images):
            raise TypeError("Every subject reference must be a PIL image.")
        if len(subject_images) > MAX_SUBJECT_REFERENCE_IMAGES:
            raise ValueError(
                f"At most {MAX_SUBJECT_REFERENCE_IMAGES} subject reference images are supported "
                "because the background is reference image 1 and FLUX.2 Klein supports up to 4 references."
            )

        requested_subject = bool(use_subject) if use_subject is not None else bool(subject_images or mask is not None)
        full_sub_without_mask = bool(full_sub_without_mask)
        used_subject_condition = requested_subject and bool(subject_images)
        used_sub_branch = requested_subject and (mask is not None or full_sub_without_mask)
        drop_subject_ref = requested_subject and not subject_images
        experiment_mode = self.default_experiment_mode if used_sub_branch else "no_sub"

        width, height = _area_normalized_size(
            main_image.size[0],
            main_image.size[1],
            resolution=self.subject_size,
        )
        guidance = self.default_guidance_scale if guidance_scale is None else float(guidance_scale)
        guidance_rescale = float(guidance_rescale)
        if not math.isfinite(guidance_rescale) or not 0.0 <= guidance_rescale <= 1.0:
            raise ValueError(f"guidance_rescale must be finite and within [0, 1], got {guidance_rescale}.")
        use_cfg_zero_star = bool(use_cfg_zero_star)
        cfg_zero_star_zero_init_steps = int(cfg_zero_star_zero_init_steps)
        if not 0 <= cfg_zero_star_zero_init_steps <= self.num_inference_steps:
            raise ValueError(
                "cfg_zero_star_zero_init_steps must be within "
                f"[0, {self.num_inference_steps}], got {cfg_zero_star_zero_init_steps}."
            )
        generator = _generator_for(self.device, seed)

        prompt_embeds, text_ids = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=self.text_encoder_out_layers,
        )
        negative_prompt_embeds, negative_text_ids = self.pipe.encode_prompt(
            prompt="",
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=self.text_encoder_out_layers,
        )

        main_pixels = image_to_tensor(main_image, width=width, height=height).to(
            self.device, self.pipe.vae.dtype
        )
        main_latents = self.pipe._encode_vae_image(main_pixels, generator=generator)
        packed_main = self.pipe._pack_latents(main_latents)

        selection = None
        sub_latents = None
        sub_ids = None
        mask_tensor = None
        subject_sizes: list[dict[str, int]] = []
        if used_sub_branch:
            if mask is None:
                mask_tensor = torch.ones((height, width), device=self.device, dtype=torch.float32)
            else:
                mask_tensor = mask_to_tensor(mask, width=width, height=height).to(self.device)
        if used_subject_condition:
            condition_latents = [main_latents]
            packed_conditions = [packed_main]
            for subject_reference in subject_images:
                subject_width, subject_height = _area_normalized_size(
                    subject_reference.size[0],
                    subject_reference.size[1],
                    resolution=self.subject_size,
                )
                subject_pixels = image_to_tensor(
                    subject_reference,
                    width=subject_width,
                    height=subject_height,
                ).to(self.device, self.pipe.vae.dtype)
                subject_latents = self.pipe._encode_vae_image(subject_pixels, generator=generator)
                condition_latents.append(subject_latents)
                packed_conditions.append(self.pipe._pack_latents(subject_latents))
                subject_sizes.append({"width": subject_width, "height": subject_height})

            packed_cond = torch.cat(packed_conditions, dim=1)
            cond_ids = build_condition_ids(
                condition_latents,
                start_t=DEFAULT_COND_T_START,
                stride=DEFAULT_COND_T_STRIDE,
                device=self.device,
            )
        else:
            packed_cond = packed_main
            cond_ids = build_condition_ids(
                [main_latents],
                start_t=DEFAULT_COND_T_START,
                stride=DEFAULT_COND_T_STRIDE,
                device=self.device,
            )
        if used_sub_branch:
            selection, sub_latents, sub_ids = build_sparse_sub_branch(
                main_latents,
                mask_tensor,
                sub_region_mode=self.sub_region_mode,
                sub_t_coord=DEFAULT_SUB_T,
                use_sparse_sub_branch=self.use_sparse_sub_branch,
                pe_exchange_region=self.pe_exchange_region,
                generator=generator,
            )
            selection = coerce_sparse_selection(selection, experiment_mode=experiment_mode)
            if selection.crop_bounds is None:
                raise ValueError("The mask produced no valid sparse crop bounds.")
            if sub_id_spatial_mode == "global":
                # `build_sparse_sub_branch` returns crop-local sub IDs (H/W relative to the
                # bbox top-left). "global" shifts them by the bbox origin so the sparse sub
                # tokens carry main-image absolute coordinates; "local" keeps the crop-local
                # IDs (train-consistent). Previously this branch tested the stale value
                # "absolute" (not in VALID_SUB_ID_SPATIAL_MODES), so the shift never fired and
                # "global" silently behaved identically to "local".
                y1, y2, x1, x2 = (int(x) for x in selection.crop_bounds)
                sub_ids = sub_ids.clone()
                sub_ids[..., 1] += y1
                sub_ids[..., 2] += x1
        num_channels_latents = self.pipe.transformer.config.in_channels // 4
        target_latents, target_ids = self.pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=self.device,
            generator=generator,
            latents=None,
        )

        sigmas = np.linspace(1.0, 1.0 / self.num_inference_steps, self.num_inference_steps)
        if getattr(self.pipe.scheduler.config, "use_flow_sigmas", False):
            sigmas = None
        mu = compute_empirical_mu(
            image_seq_len=target_latents.shape[1],
            num_steps=self.num_inference_steps,
        )
        timesteps, _ = retrieve_timesteps(
            self.pipe.scheduler,
            self.num_inference_steps,
            self.device,
            sigmas=sigmas,
            mu=mu,
        )
        sub_scheduler = copy.deepcopy(self.pipe.scheduler) if sub_latents is not None else None
        self.pipe.scheduler.set_begin_index(0)
        if sub_scheduler is not None:
            sub_scheduler.set_begin_index(0)
        total_target_tokens = target_latents.shape[1]
        has_guidance_embeds = getattr(self.pipe.transformer.config, "guidance_embeds", False)
        collect_ste_gates = selection is not None and hasattr(self.pipe.transformer, "start_ste_gate_collection")
        if collect_ste_gates:
            self.pipe.transformer.start_ste_gate_collection(
                capture_raw=save_raw_ste,
                branches=("cond", "uncond") if save_raw_ste else ("cond",),
            )
        collect_cross_attention = (
            selection is not None
            and cross_attention_layers is not None
            and len(tuple(cross_attention_layers)) > 0
            and hasattr(self.pipe.transformer, "start_cross_attention_collection")
        )
        self.last_cross_attention_collection = None
        if hasattr(self.pipe.transformer, "clear_pe_gate_overrides"):
            self.pipe.transformer.clear_pe_gate_overrides()
            if force_gate_overrides:
                self.pipe.transformer.set_pe_gate_overrides(force_gate_overrides)
        if collect_cross_attention:
            self.pipe.transformer.start_cross_attention_collection(
                layers=tuple(int(layer_idx) for layer_idx in cross_attention_layers),
                branches=tuple(cross_attention_branches),
            )

        scheduler_sigmas = getattr(self.pipe.scheduler, "sigmas", None)
        denoising_schedule: list[dict[str, float | int | None]] = []

        for step_index, timestep_value in enumerate(timesteps):
            sigma = None
            if scheduler_sigmas is not None and step_index < len(scheduler_sigmas):
                sigma = float(scheduler_sigmas[step_index].detach().float().item())
            denoising_schedule.append(
                {
                    "step_index": int(step_index),
                    "timestep": float(timestep_value.detach().float().item()),
                    "sigma": sigma,
                }
            )
            timestep = timestep_value.expand(1).to(target_latents.dtype)
            hidden_parts = [target_latents.to(self.torch_dtype)]
            img_id_parts = [target_ids]
            if sub_latents is not None and sub_ids is not None:
                hidden_parts.append(sub_latents.to(self.torch_dtype))
                img_id_parts.append(sub_ids)
            hidden_parts.append(packed_cond.to(self.torch_dtype))
            img_id_parts.append(cond_ids)
            hidden_states = torch.cat(hidden_parts, dim=1)
            img_ids = torch.cat(img_id_parts, dim=1)
            guidance_tensor = None
            if has_guidance_embeds:
                guidance_tensor = torch.full([1], guidance, device=self.device, dtype=target_latents.dtype)

            model_kwargs = {
                "hidden_states": hidden_states,
                "timestep": timestep / 1000,
                "guidance": guidance_tensor,
                "encoder_hidden_states": prompt_embeds,
                "txt_ids": text_ids,
                "img_ids": img_ids,
                "joint_attention_kwargs": None,
                "return_dict": False,
                "num_cond_tokens": int(packed_cond.shape[1]),
            }
            if selection is not None:
                model_kwargs["pe_sparse_token_selection"] = selection
                model_kwargs["pe_ref_t_coord"] = DEFAULT_SUB_T
                model_kwargs["pe_main_t_coord"] = 0
            if collect_ste_gates:
                self.pipe.transformer.set_ste_gate_collection_branch("cond")
            if collect_cross_attention:
                self.pipe.transformer.set_cross_attention_collection_branch("cond")
            with self.pipe.transformer.cache_context("cond"):
                pred = self.pipe.transformer(**model_kwargs)[0]
            if collect_ste_gates:
                self.pipe.transformer.set_ste_gate_collection_branch("uncond" if save_raw_ste else None)
            if collect_cross_attention:
                self.pipe.transformer.set_cross_attention_collection_branch(
                    "uncond" if "uncond" in cross_attention_branches else None
                )
            negative_kwargs = dict(model_kwargs)
            negative_kwargs["encoder_hidden_states"] = negative_prompt_embeds
            negative_kwargs["txt_ids"] = negative_text_ids
            with self.pipe.transformer.cache_context("uncond"):
                negative_pred = self.pipe.transformer(**negative_kwargs)[0]
            if collect_ste_gates:
                self.pipe.transformer.pause_ste_gate_collection()
            if collect_cross_attention:
                self.pipe.transformer.pause_cross_attention_collection()
            conditional_pred = pred
            target_slice = slice(0, total_target_tokens)
            sub_slice = (
                slice(total_target_tokens, total_target_tokens + sub_latents.shape[1])
                if sub_latents is not None
                else None
            )
            if use_cfg_zero_star and step_index < cfg_zero_star_zero_init_steps:
                pred = torch.zeros_like(conditional_pred)
            elif use_cfg_zero_star:
                pred = negative_pred + guidance * (conditional_pred - negative_pred)
                pred[:, target_slice, :] = apply_cfg_zero_star_guidance(
                    conditional_pred[:, target_slice, :],
                    negative_pred[:, target_slice, :],
                    guidance,
                )
                if sub_slice is not None:
                    pred[:, sub_slice, :] = apply_cfg_zero_star_guidance(
                        conditional_pred[:, sub_slice, :],
                        negative_pred[:, sub_slice, :],
                        guidance,
                    )
            else:
                pred = negative_pred + guidance * (conditional_pred - negative_pred)

            if guidance_rescale > 0.0 and not (
                use_cfg_zero_star and step_index < cfg_zero_star_zero_init_steps
            ):
                pred[:, target_slice, :] = rescale_cfg_prediction(
                    pred[:, target_slice, :], conditional_pred[:, target_slice, :], guidance_rescale
                )
                if sub_slice is not None:
                    pred[:, sub_slice, :] = rescale_cfg_prediction(
                        pred[:, sub_slice, :], conditional_pred[:, sub_slice, :], guidance_rescale
                    )

            target_latents = self.pipe.scheduler.step(
                pred[:, :total_target_tokens, :],
                timestep_value,
                target_latents,
                return_dict=False,
            )[0]
            if sub_latents is not None:
                assert sub_scheduler is not None
                sub_latents = sub_scheduler.step(
                    pred[:, total_target_tokens : total_target_tokens + sub_latents.shape[1], :],
                    timestep_value,
                    sub_latents,
                    return_dict=False,
                )[0]
            if progress_callback is not None:
                progress_callback(step_index + 1, len(timesteps))

        gate_collection = self.pipe.transformer.pop_ste_gate_collection() if collect_ste_gates else None
        cross_attention_collection = (
            self.pipe.transformer.pop_cross_attention_collection() if collect_cross_attention else None
        )
        self.last_cross_attention_collection = cross_attention_collection
        if hasattr(self.pipe.transformer, "clear_pe_gate_overrides"):
            self.pipe.transformer.clear_pe_gate_overrides()

        main_out = self._decode_grid(self._unpack_target(target_latents, target_ids))

        sub_out = None
        crop_bounds = None
        sub_token_count = 0
        crop_token_count = 0
        if selection is not None and sub_latents is not None and selection.crop_bounds is not None:
            y1, y2, x1, x2 = selection.crop_bounds
            crop_h = int(y2 - y1)
            crop_w = int(x2 - x1)
            token_w = width // 16
            main_idx = selection.main_token_indices.to(self.device, dtype=torch.long)
            local_idx = ((main_idx // token_w - y1) * crop_w + (main_idx % token_w - x1)).long()
            white_rect = self.pipe._encode_vae_image(
                torch.ones(
                    1,
                    3,
                    crop_h * 16,
                    crop_w * 16,
                    device=self.device,
                    dtype=self.pipe.vae.dtype,
                ),
                generator=None,
            )
            dense_sub = self.pipe._pack_latents(white_rect).clone()
            dense_sub[0, local_idx] = sub_latents[0].to(dense_sub.dtype)
            sub_out = self._decode_grid(dense_sub.view(1, crop_h, crop_w, -1).permute(0, 3, 1, 2))
            crop_bounds = [int(y1), int(y2), int(x1), int(x2)]
            sub_token_count = int(sub_latents.shape[1])
            crop_token_count = int(crop_h * crop_w)

        ste_masks = self._save_ste_masks(
            gate_collection,
            width=width,
            height=height,
            denoising_schedule=denoising_schedule,
            sub_id_spatial_mode=sub_id_spatial_mode,
        )

        metadata = {
            **self.load_metadata,
            "prompt": prompt,
            "experiment_mode": experiment_mode,
            "used_subject": requested_subject,
            "used_subject_condition": used_subject_condition,
            "used_sub_branch": used_sub_branch,
            "mask_used": mask is not None,
            "full_sub_without_mask": bool(used_sub_branch and mask is None),
            "sub_mask_source": (
                "input_mask" if mask is not None else "full_image" if used_sub_branch else None
            ),
            "drop_subject_reference": drop_subject_ref,
            "seed": int(seed),
            "guidance_scale": guidance,
            "guidance_rescale": guidance_rescale,
            "use_cfg_zero_star": use_cfg_zero_star,
            "cfg_zero_star_zero_init_steps": cfg_zero_star_zero_init_steps,
            "num_inference_steps": self.num_inference_steps,
            "snapped_width": width,
            "snapped_height": height,
            "subject_size": self.subject_size,
            "subject_image_count": len(subject_images),
            "subject_sizes": subject_sizes,
            "subject_width": subject_sizes[0]["width"] if subject_sizes else None,
            "subject_height": subject_sizes[0]["height"] if subject_sizes else None,
            "sub_token_count": sub_token_count,
            "crop_token_count": crop_token_count,
            "crop_bounds": crop_bounds,
            "sub_region_mode": self.sub_region_mode,
            "use_sparse_sub_branch": self.use_sparse_sub_branch,
            "pe_exchange_region": self.pe_exchange_region,
            "sub_id_spatial_mode": sub_id_spatial_mode,
            "raw_ste_saved": bool(save_raw_ste),
            "ste_masks": ste_masks,
            "forced_gate_overrides": ({int(k): float(v) for k, v in force_gate_overrides.items()} if force_gate_overrides else None),
            "cross_attention_layers": [int(layer_idx) for layer_idx in cross_attention_layers] if cross_attention_layers else [],
            "cross_attention_branches": list(cross_attention_branches) if collect_cross_attention else [],
        }
        return GenerationResult(main_image=main_out, sub_image=sub_out, metadata=metadata)
