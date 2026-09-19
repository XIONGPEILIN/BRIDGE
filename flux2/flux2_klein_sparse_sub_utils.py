from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageOps

from qwen_pe_exchange_sparse_model import (
    SparseTokenSelection,
    build_sparse_token_selection_from_mask,
    select_sparse_tokens,
)


REPO_ROOT = Path(__file__).resolve().parent
LOCAL_DIFFUSERS_SRC = REPO_ROOT / "diffusers" / "src"
if LOCAL_DIFFUSERS_SRC.exists():
    import sys

    sys.path.insert(0, str(LOCAL_DIFFUSERS_SRC))

from diffusers import Flux2KleinPipeline  # noqa: E402


@dataclass
class Flux2SparseSubLatents:
    sparse_token_selection: SparseTokenSelection
    packed_latents: torch.Tensor
    latent_ids: torch.Tensor
    sparse_x0: torch.Tensor
    sparse_ids: torch.Tensor
    noise: Optional[torch.Tensor] = None
    noisy_sparse_x: Optional[torch.Tensor] = None


def load_mask_tensor(
    mask: str | Path | Image.Image | torch.Tensor,
    resolution: Optional[int] = None,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        mask_tensor = mask.to(dtype=torch.float32)
        if float(mask_tensor.max().item()) > 1.0:
            mask_tensor = (mask_tensor > 128.0).to(dtype=torch.float32)
        else:
            mask_tensor = (mask_tensor > 0.5).to(dtype=torch.float32)
        return mask_tensor.to(device=device) if device is not None else mask_tensor

    if isinstance(mask, (str, Path)):
        image = Image.open(mask)
    elif isinstance(mask, Image.Image):
        image = mask
    else:
        raise TypeError(f"Unsupported mask type: {type(mask)}")

    image = ImageOps.exif_transpose(image).convert("L")
    if resolution is not None:
        image = image.resize((resolution, resolution), Image.Resampling.NEAREST)

    mask_tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    mask_tensor = mask_tensor.view(image.height, image.width).to(torch.float32)
    mask_tensor = (mask_tensor > 128.0).to(torch.float32)
    return mask_tensor.to(device=device) if device is not None else mask_tensor


def build_sparse_sub_latents(
    latents: torch.Tensor,
    mask: str | Path | Image.Image | torch.Tensor,
    *,
    latent_ids: Optional[torch.Tensor] = None,
    resolution: Optional[int] = None,
    downsample_factor: int = 16,
    threshold: float = 0.5,
) -> Flux2SparseSubLatents:
    """
    Build sparse FLUX.2 Klein sub tokens from dense VAE latents.

    This follows the intended training semantics:
    1. define sparse sub x0 from the mask-selected token set
    2. add noise on sparse sub x0 afterwards
    """

    if latents.dim() != 4:
        raise ValueError(f"Expected latents shape [B, C, H, W], got {tuple(latents.shape)}")

    mask_tensor = load_mask_tensor(mask, resolution=resolution, device=latents.device)
    sparse_token_selection = build_sparse_token_selection_from_mask(
        mask_tensor,
        downsample_factor=downsample_factor,
        threshold=threshold,
    )

    packed_latents = Flux2KleinPipeline._pack_latents(latents)
    if latent_ids is None:
        latent_ids = Flux2KleinPipeline._prepare_latent_ids(latents).to(device=latents.device)
    else:
        latent_ids = latent_ids.to(device=latents.device)

    if sparse_token_selection.crop_token_indices is None or sparse_token_selection.crop_token_indices.numel() == 0:
        raise ValueError("The selected mask region produced zero crop tokens.")

    sparse_x0 = select_sparse_tokens(packed_latents, sparse_token_selection.crop_token_indices)
    sparse_ids = select_sparse_tokens(latent_ids, sparse_token_selection.crop_token_indices)

    return Flux2SparseSubLatents(
        sparse_token_selection=sparse_token_selection,
        packed_latents=packed_latents,
        latent_ids=latent_ids,
        sparse_x0=sparse_x0,
        sparse_ids=sparse_ids,
    )


def expand_sigmas_for_tokens(
    sigmas: torch.Tensor | float,
    x: torch.Tensor,
) -> torch.Tensor:
    sigmas = torch.as_tensor(sigmas, device=x.device, dtype=x.dtype)
    if sigmas.dim() == 0:
        return sigmas.view(1, 1, 1)
    if sigmas.dim() == 1:
        if sigmas.shape[0] != x.shape[0]:
            raise ValueError(f"Sigma batch {sigmas.shape[0]} must match x batch {x.shape[0]}.")
        return sigmas.view(-1, 1, 1)
    if sigmas.dim() == 3:
        return sigmas.to(device=x.device, dtype=x.dtype)
    raise ValueError(f"Unsupported sigma shape {tuple(sigmas.shape)}.")


def add_flow_matching_noise(
    sparse_x0: torch.Tensor,
    sigmas: torch.Tensor | float,
    noise: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply FLUX.2 flow-matching noise on sparse tokens.

    z_t = (1 - sigma) * x_0 + sigma * eps
    """

    if sparse_x0.dim() != 3:
        raise ValueError(f"Expected sparse_x0 shape [B, K, C], got {tuple(sparse_x0.shape)}")

    if noise is None:
        noise = torch.randn_like(sparse_x0)
    else:
        noise = noise.to(device=sparse_x0.device, dtype=sparse_x0.dtype)
        if noise.shape != sparse_x0.shape:
            raise ValueError(f"Noise shape {tuple(noise.shape)} must match sparse_x0 {tuple(sparse_x0.shape)}.")

    sigma_view = expand_sigmas_for_tokens(sigmas, sparse_x0)
    noisy_sparse_x = (1.0 - sigma_view) * sparse_x0 + sigma_view * noise
    return noisy_sparse_x, noise


def build_noisy_sparse_sub_latents(
    latents: torch.Tensor,
    mask: str | Path | Image.Image | torch.Tensor,
    *,
    sigmas: torch.Tensor | float,
    noise: Optional[torch.Tensor] = None,
    latent_ids: Optional[torch.Tensor] = None,
    resolution: Optional[int] = None,
    downsample_factor: int = 16,
    threshold: float = 0.5,
) -> Flux2SparseSubLatents:
    sparse = build_sparse_sub_latents(
        latents,
        mask,
        latent_ids=latent_ids,
        resolution=resolution,
        downsample_factor=downsample_factor,
        threshold=threshold,
    )
    noisy_sparse_x, used_noise = add_flow_matching_noise(sparse.sparse_x0, sigmas=sigmas, noise=noise)
    sparse.noisy_sparse_x = noisy_sparse_x
    sparse.noise = used_noise
    return sparse
