from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


PEType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], list[torch.Tensor]]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones((dim,)))
        else:
            self.weight = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).square().mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states.to(input_dtype)
        if self.weight is not None:
            hidden_states = hidden_states * self.weight
        return hidden_states


@dataclass
class PEExchangeOutput:
    pe_encoding: PEType
    token_gate: Optional[torch.Tensor]
    swapped_pe: Optional[torch.Tensor]
    base_pe: torch.Tensor
    aux: dict[str, Any]


# Share the SAME SparseTokenSelection class as the hard model. The training pipeline builds
# selections from `qwen_pe_exchange_sparse_model`, so defining a separate (structurally
# identical) dataclass here made `isinstance(sel, SparseTokenSelection)` fail at training
# time — the soft model saw a hard-model instance of a different class and fell through to
# the tensor branch (`TypeError: 'SparseTokenSelection' object cannot be interpreted as an
# integer`). Importing it unifies the class across the whole pipeline.
from qwen_pe_exchange_sparse_model import (  # noqa: E402
    SparseTokenSelection,
    _normalize_src_key_padding_mask,
)


def _normalize_mask_tensor(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    if mask.dim() == 4:
        pass
    elif mask.dim() not in (2, 3, 4):
        raise ValueError(f"Expected mask dims 2/3/4, got {mask.dim()}.")

    mask = mask.to(dtype=torch.float32)
    if float(mask.max().item()) > 1.0:
        mask = (mask > 128.0).to(dtype=torch.float32)
    else:
        mask = (mask > 0.5).to(dtype=torch.float32)
    return mask


def build_sparse_token_selection_from_mask(
    mask: torch.Tensor,
    downsample_factor: int = 16,
    threshold: float = 0.5,
) -> SparseTokenSelection:
    """
    Convert a pixel-space mask into sparse token indices using outward rounding.

    A token is selected if any pixel inside its `downsample_factor x downsample_factor`
    block is white/non-zero, which matches the user's "向外取整" requirement.
    """

    mask = _normalize_mask_tensor(mask).to(dtype=torch.float32)
    pooled = F.max_pool2d(
        mask,
        kernel_size=downsample_factor,
        stride=downsample_factor,
        ceil_mode=True,
    )
    token_mask = pooled[0, 0] > threshold
    token_coords = token_mask.nonzero(as_tuple=False)

    if token_coords.numel() == 0:
        empty = torch.zeros((0,), dtype=torch.long, device=token_mask.device)
        return SparseTokenSelection(
            main_token_indices=empty,
            ref_token_indices=empty,
            token_mask=token_mask,
            token_coords=token_coords,
            crop_token_indices=empty,
            crop_token_coords=token_coords,
            crop_bounds=None,
        )

    token_h, token_w = token_mask.shape
    y1 = int(token_coords[:, 0].min().item())
    y2 = int(token_coords[:, 0].max().item()) + 1
    x1 = int(token_coords[:, 1].min().item())
    x2 = int(token_coords[:, 1].max().item()) + 1

    ys = torch.arange(y1, y2, device=token_mask.device, dtype=torch.long)
    xs = torch.arange(x1, x2, device=token_mask.device, dtype=torch.long)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    crop_token_coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=1)
    crop_token_indices = (crop_token_coords[:, 0] * token_w + crop_token_coords[:, 1]).to(torch.long)

    main_token_indices = (token_coords[:, 0] * token_w + token_coords[:, 1]).to(torch.long)
    local_coords = token_coords.clone()
    local_coords[:, 0] -= y1
    local_coords[:, 1] -= x1
    crop_w = x2 - x1
    ref_token_indices = (local_coords[:, 0] * crop_w + local_coords[:, 1]).to(torch.long)

    return SparseTokenSelection(
        main_token_indices=main_token_indices,
        ref_token_indices=ref_token_indices,
        token_mask=token_mask,
        token_coords=token_coords,
        crop_token_indices=crop_token_indices,
        crop_token_coords=crop_token_coords,
        crop_bounds=(y1, y2, x1, x2),
    )


def select_sparse_tokens(
    packed_tokens: torch.Tensor,
    sparse_token_selection: SparseTokenSelection | torch.Tensor,
    use_ref_indices: bool = False,
) -> torch.Tensor:
    """
    Select only the sparse white-region tokens from a packed token tensor.

    This is the scheme-A companion utility:
    - input packed main/ref tokens: [B, L, C] or [L, C]
    - output sparse sub tokens only for selected white positions
    """

    if isinstance(sparse_token_selection, SparseTokenSelection):
        indices = sparse_token_selection.ref_token_indices if use_ref_indices else sparse_token_selection.main_token_indices
    else:
        indices = sparse_token_selection

    indices = torch.as_tensor(indices, device=packed_tokens.device, dtype=torch.long).view(-1)

    if packed_tokens.dim() == 2:
        return packed_tokens.index_select(0, indices)
    if packed_tokens.dim() == 3:
        return packed_tokens.index_select(1, indices)
    raise ValueError(f"Expected packed token dims 2 or 3, got {packed_tokens.dim()}.")


class STEBlock(nn.Module):
    def __init__(self, dim_in: int = 3072, dim: int = 128, num_heads: int = 8, encoder_layers: int = 1):
        super().__init__()
        self.proj_down = nn.Linear(dim_in, dim)
        if encoder_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim,
                batch_first=True,
                norm_first=True,
            )
            encoder_layer.norm1 = RMSNorm(dim, eps=1e-6)
            encoder_layer.norm2 = RMSNorm(dim, eps=1e-6)
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=encoder_layers,
                norm=RMSNorm(dim, eps=1e-6),
            )
        else:
            self.encoder = nn.Identity()

    def _run_encoder(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not isinstance(self.encoder, nn.TransformerEncoder):
            return self.encoder(x)

        mha_backend = getattr(torch.backends, "mha", None)
        if mha_backend is None:
            return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        old_fastpath = mha_backend.get_fastpath_enabled()
        mha_backend.set_fastpath_enabled(False)
        try:
            return self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        finally:
            mha_backend.set_fastpath_enabled(old_fastpath)

    @torch.compiler.disable(recursive=True)
    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src_key_padding_mask = _normalize_src_key_padding_mask(x, src_key_padding_mask)
        x = self.proj_down(x)
        if x.dim() == 2:
            x = x.unsqueeze(0)
            return self._run_encoder(x, src_key_padding_mask=src_key_padding_mask).squeeze(0)
        return self._run_encoder(x, src_key_padding_mask=src_key_padding_mask)


class STE(nn.Module):
    def __init__(
        self,
        dim_in: int = 3072,
        dim_out: int = 1,
        num_layers: int = 60,
        sampler: str = "vanilla_ste",
        temperature: float = 1.0,
        min_temperature: float = 0.3,
        anneal_strategy: str = "exp",
        anneal_rate: float = 3e-5,
        eval_temperature: Optional[float] = None,
        entropy_weight: float = 0.0,
        sparsity_target: Optional[float] = None,
        sparsity_weight: float = 0.0,
        hc_beta: float = 2 / 3,
        hc_gamma: float = -0.1,
        hc_zeta: float = 1.1,
        grad_scale: float = 1.0,
        head_init: str = "zero",
        head_init_std: float = 1e-3,
        return_aux: bool = False,
        encoder_layers: int = 1,
        encoder_num_heads: int = 8,
    ):
        super().__init__()
        if sampler not in ("gumbel", "bernoulli", "hard_concrete", "vanilla_ste"):
            raise ValueError(f"Unsupported sampler: {sampler}")
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.sampler = sampler
        self.return_aux = return_aux

        self.register_buffer("tau", torch.tensor(float(temperature)))
        self.tau_min = float(min_temperature)
        self.anneal_strategy = anneal_strategy
        self.anneal_rate = float(anneal_rate)
        self.eval_tau = float(eval_temperature) if eval_temperature is not None else None

        self.entropy_weight = float(entropy_weight)
        self.sparsity_target = sparsity_target
        self.sparsity_weight = float(sparsity_weight)

        self.hc_beta = float(hc_beta)
        self.hc_gamma = float(hc_gamma)
        self.hc_zeta = float(hc_zeta)
        self.grad_scale = float(grad_scale)

        self.heads = nn.ModuleList([nn.Linear(128, dim_out) for _ in range(num_layers)])
        if head_init == "zero":
            for head in self.heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        else:
            for head in self.heads:
                nn.init.normal_(head.weight, mean=0.0, std=head_init_std)
                nn.init.zeros_(head.bias)

        self.ste_blocks = nn.ModuleList(
            [
                STEBlock(dim_in=dim_in, dim=128, num_heads=encoder_num_heads, encoder_layers=encoder_layers)
                for _ in range(num_layers)
            ]
        )

    @torch.no_grad()
    def _sample_logistic_noise(self, shape, device, dtype, eps: float = 1e-6) -> torch.Tensor:
        u = torch.rand(shape, device=device, dtype=dtype).clamp_(eps, 1.0 - eps)
        return torch.log(u) - torch.log(1.0 - u)

    def _straight_through(self, y_soft: torch.Tensor, y_hard: torch.Tensor) -> torch.Tensor:
        return y_hard + self.grad_scale * (y_soft - y_soft.detach())

    def _gumbel_soft(self, logits: torch.Tensor, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = self._sample_logistic_noise(logits.shape, logits.device, logits.dtype)
        y_soft = torch.sigmoid((logits + g) / tau)
        return y_soft, y_soft

    def _bernoulli_soft(self, logits: torch.Tensor, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.sigmoid(logits / tau)
        return probs, probs

    def _hard_concrete_soft(self, logits: torch.Tensor, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
        s = torch.sigmoid((logits + torch.log(u) - torch.log(1 - u)) / self.hc_beta)
        z_tilde = s * (self.hc_zeta - self.hc_gamma) + self.hc_gamma
        z = z_tilde.clamp_(0.0, 1.0)
        probs = torch.sigmoid(logits / tau)
        return z, probs

    def _vanilla_soft(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_soft = torch.sigmoid(logits)
        return y_soft, y_soft

    def anneal_temperature(self, steps: int = 1) -> float:
        if self.anneal_strategy == "none" or self.anneal_rate <= 0:
            return float(self.tau)
        if self.anneal_strategy == "exp":
            with torch.no_grad():
                self.tau.mul_(math.exp(-self.anneal_rate * steps)).clamp_(min=self.tau_min)
        elif self.anneal_strategy == "linear":
            with torch.no_grad():
                self.tau.sub_(self.anneal_rate * steps).clamp_(min=self.tau_min)
        return float(self.tau)

    @torch.compiler.disable(recursive=True)
    def forward(
        self,
        h: torch.Tensor,
        layer_idx: int,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        src_key_padding_mask = _normalize_src_key_padding_mask(h, src_key_padding_mask)
        h_enc = self.ste_blocks[layer_idx](h, src_key_padding_mask=src_key_padding_mask)
        logits = self.heads[layer_idx](h_enc)
        tau = self.tau if self.training or self.eval_tau is None else torch.tensor(
            self.eval_tau, device=h.device, dtype=h.dtype
        )

        if self.training:
            if self.sampler == "gumbel":
                mask, probs = self._gumbel_soft(logits, tau)
            elif self.sampler == "bernoulli":
                mask, probs = self._bernoulli_soft(logits, tau)
            elif self.sampler == "hard_concrete":
                mask, probs = self._hard_concrete_soft(logits, tau)
            else:
                mask, probs = self._vanilla_soft(logits)
        else:
            probs = torch.sigmoid(logits / tau)
            if self.sampler == "hard_concrete":
                mask = (probs * (self.hc_zeta - self.hc_gamma) + self.hc_gamma).clamp_(0.0, 1.0)
            else:
                mask = probs

        reg_loss = torch.as_tensor(0.0, device=h.device, dtype=h.dtype)
        if self.training and (
            self.entropy_weight > 0.0 or (self.sparsity_target is not None and self.sparsity_weight > 0.0)
        ):
            regularized_probs = probs
            if src_key_padding_mask is not None:
                valid = (~src_key_padding_mask).unsqueeze(-1).expand_as(probs)
                regularized_probs = probs.masked_select(valid)
                if regularized_probs.numel() == 0:
                    raise ValueError("STE src_key_padding_mask cannot mask every token.")
            p = regularized_probs.clamp(1e-6, 1 - 1e-6)
            if self.entropy_weight > 0.0:
                ent = -(p * p.log() + (1 - p) * (1 - p).log())
                reg_loss = reg_loss + ent.mean() * self.entropy_weight
            if self.sparsity_target is not None and self.sparsity_weight > 0.0:
                mean_p = p.mean()
                target = torch.tensor(self.sparsity_target, device=p.device, dtype=p.dtype)
                reg_loss = reg_loss + F.mse_loss(mean_p, target) * self.sparsity_weight

        aux = {
            "probs": probs,
            "logits": logits,
            "reg_loss": reg_loss,
            "temperature": torch.as_tensor(float(tau), device=h.device, dtype=h.dtype),
            "soft_gate": mask,
        }
        return mask, (aux if self.return_aux else {})


class QwenPEExchangeModel(nn.Module):
    def __init__(
        self,
        latent_channels: int = 16,
        patch_size: int = 2,
        hidden_dim: int = 3072,
        pe_dim: int = 64,
        num_layers: int = 60,
        sampler: str = "vanilla_ste",
        temperature: float = 1.0,
        min_temperature: float = 0.3,
        anneal_strategy: str = "exp",
        anneal_rate: float = 3e-5,
        eval_temperature: Optional[float] = None,
        entropy_weight: float = 0.0,
        sparsity_target: Optional[float] = None,
        sparsity_weight: float = 0.0,
        hc_beta: float = 2 / 3,
        hc_gamma: float = -0.1,
        hc_zeta: float = 1.1,
        grad_scale: float = 1.0,
        head_init: str = "zero",
        head_init_std: float = 1e-3,
        return_aux: bool = True,
        encoder_layers: int = 1,
        encoder_num_heads: int = 8,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.pe_dim = pe_dim
        self.token_dim = latent_channels * patch_size * patch_size

        self.img_in = nn.Linear(self.token_dim, hidden_dim)
        self.ste = STE(
            dim_in=hidden_dim,
            dim_out=1,
            num_layers=num_layers,
            sampler=sampler,
            temperature=temperature,
            min_temperature=min_temperature,
            anneal_strategy=anneal_strategy,
            anneal_rate=anneal_rate,
            eval_temperature=eval_temperature,
            entropy_weight=entropy_weight,
            sparsity_target=sparsity_target,
            sparsity_weight=sparsity_weight,
            hc_beta=hc_beta,
            hc_gamma=hc_gamma,
            hc_zeta=hc_zeta,
            grad_scale=grad_scale,
            head_init=head_init,
            head_init_std=head_init_std,
            return_aux=return_aux,
            encoder_layers=encoder_layers,
            encoder_num_heads=encoder_num_heads,
        )

    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.dim() != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(latents.shape)}.")
        batch, channels, height, width = latents.shape
        if channels != self.latent_channels:
            raise ValueError(f"Expected {self.latent_channels} channels, got {channels}.")
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"Latent size {(height, width)} must be divisible by patch_size={self.patch_size}."
            )

        h_tokens = height // self.patch_size
        w_tokens = width // self.patch_size
        latents = latents.reshape(
            batch,
            channels,
            h_tokens,
            self.patch_size,
            w_tokens,
            self.patch_size,
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5).contiguous()
        return latents.reshape(batch, h_tokens * w_tokens, self.token_dim)

    def _split_pe(self, pe_encoding: PEType) -> tuple[tuple[torch.Tensor, ...], str]:
        if isinstance(pe_encoding, torch.Tensor):
            return (pe_encoding,), "tensor"
        if isinstance(pe_encoding, tuple):
            if len(pe_encoding) != 2:
                raise ValueError("PE tuple input must be a RoPE pair `(cos, sin)`.")
            return tuple(pe_encoding), "tuple"
        if isinstance(pe_encoding, list):
            if len(pe_encoding) != 2:
                raise ValueError("PE list input must be a RoPE pair `[cos, sin]`.")
            return tuple(pe_encoding), "list"
        raise TypeError(f"Unsupported pe_encoding type: {type(pe_encoding)}")

    def _merge_pe(self, pe_components: tuple[torch.Tensor, ...], kind: str) -> PEType:
        if kind == "tensor":
            return pe_components[0]
        if kind == "tuple":
            return tuple(pe_components)
        return list(pe_components)

    def _build_swapped_pe(
        self,
        base_img_pe: torch.Tensor,
        subyx: tuple[int, int, int, int],
        main_patch_hw: tuple[int, int],
        sub_patch_hw: tuple[int, int],
    ) -> torch.Tensor:
        y1, y2, x1, x2 = subyx
        main_h, main_w = main_patch_hw
        sub_h, sub_w = sub_patch_hw
        main_len = main_h * main_w
        sub_len = sub_h * sub_w

        if base_img_pe.dim() != 2:
            raise ValueError(f"Expected base image PE to be [seq, dim], got {tuple(base_img_pe.shape)}.")
        if base_img_pe.shape[0] < main_len + sub_len:
            raise ValueError(
                f"PE length {base_img_pe.shape[0]} is smaller than main+sub length {main_len + sub_len}."
            )
        if (y2 - y1) != sub_h or (x2 - x1) != sub_w:
            raise ValueError(
                f"subyx region {(y1, y2, x1, x2)} does not match sub patch size {(sub_h, sub_w)}."
            )

        main_pe = base_img_pe[:main_len].reshape(main_h, main_w, -1)
        sub_pe = main_pe[y1:y2, x1:x2, :].reshape(sub_len, -1)
        return torch.cat([main_pe.reshape(main_len, -1), sub_pe], dim=0)

    def _expand_gate(self, token_gate: torch.Tensor, valid_len: int, pe_dim: int) -> torch.Tensor:
        if token_gate.dim() == 2:
            token_gate = token_gate.unsqueeze(-1)
        if token_gate.dim() != 3:
            raise ValueError(f"Expected token gate to be [B, L, 1] or [L, 1], got {tuple(token_gate.shape)}.")

        token_gate = token_gate[:, :valid_len, :]
        if token_gate.shape[-1] == 1:
            token_gate = token_gate.expand(-1, -1, pe_dim)
        if token_gate.shape[0] == 1:
            return token_gate[0]
        return token_gate.mean(dim=0)

    def forward(
        self,
        noise_latents: torch.Tensor,
        image_latents: Optional[torch.Tensor],
        pe_encoding: PEType,
        subyx: Optional[tuple[int, int, int, int]],
        layer_idx: int,
        return_dict: bool = True,
        ste_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Union[PEExchangeOutput, PEType]:
        pe_components, pe_kind = self._split_pe(pe_encoding)
        base_img_freqs = pe_components[0]

        if image_latents is None or subyx is None:
            output = PEExchangeOutput(
                pe_encoding=self._merge_pe(pe_components, pe_kind),
                token_gate=None,
                swapped_pe=None,
                base_pe=base_img_freqs,
                aux={"reason": "image_latents or subyx is None"},
            )
            return output if return_dict else output.pe_encoding

        main_tokens = self._pack_latents(noise_latents)
        sub_tokens = self._pack_latents(image_latents)
        image_tokens = torch.cat([main_tokens, sub_tokens], dim=1)
        hidden_states = self.img_in(image_tokens)

        token_gate, aux = self.ste(
            hidden_states,
            layer_idx=layer_idx,
            src_key_padding_mask=ste_key_padding_mask,
        )

        main_patch_hw = (
            noise_latents.shape[2] // self.patch_size,
            noise_latents.shape[3] // self.patch_size,
        )
        sub_patch_hw = (
            image_latents.shape[2] // self.patch_size,
            image_latents.shape[3] // self.patch_size,
        )
        swapped_components = tuple(
            self._build_swapped_pe(component, subyx, main_patch_hw, sub_patch_hw)
            for component in pe_components
        )
        swapped_pe = swapped_components[0]
        expanded_gate = self._expand_gate(token_gate, swapped_pe.shape[0], swapped_pe.shape[1])
        mixed_components = []
        for base_component, swapped_component in zip(pe_components, swapped_components):
            mixed_component = (
                swapped_component * (1 - expanded_gate)
                + base_component[: swapped_component.shape[0], :] * expanded_gate
            )
            updated_component = base_component.clone()
            updated_component[: mixed_component.shape[0], :] = mixed_component
            mixed_components.append(updated_component)
        merged_pe = self._merge_pe(tuple(mixed_components), pe_kind)

        output = PEExchangeOutput(
            pe_encoding=merged_pe,
            token_gate=expanded_gate,
            swapped_pe=swapped_pe,
            base_pe=base_img_freqs,
            aux={
                "raw_gate": token_gate,
                "ste_aux": aux,
                "main_patch_hw": main_patch_hw,
                "sub_patch_hw": sub_patch_hw,
                "main_seq_len": main_tokens.shape[1],
                "sub_seq_len": sub_tokens.shape[1],
                "pe_component_count": len(pe_components),
            },
        )
        return output if return_dict else output.pe_encoding


class PackedSparsePEExchangeModel(nn.Module):
    def __init__(
        self,
        dim_in: int = 3072,
        pe_dim: int = 64,
        num_layers: int = 60,
        sampler: str = "vanilla_ste",
        temperature: float = 1.0,
        min_temperature: float = 0.3,
        anneal_strategy: str = "exp",
        anneal_rate: float = 3e-5,
        eval_temperature: Optional[float] = None,
        entropy_weight: float = 0.0,
        sparsity_target: Optional[float] = None,
        sparsity_weight: float = 0.0,
        hc_beta: float = 2 / 3,
        hc_gamma: float = -0.1,
        hc_zeta: float = 1.1,
        grad_scale: float = 1.0,
        head_init: str = "zero",
        head_init_std: float = 1e-3,
        return_aux: bool = True,
        encoder_layers: int = 1,
        encoder_num_heads: int = 8,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.pe_dim = pe_dim
        self.ste = STE(
            dim_in=dim_in,
            dim_out=1,
            num_layers=num_layers,
            sampler=sampler,
            temperature=temperature,
            min_temperature=min_temperature,
            anneal_strategy=anneal_strategy,
            anneal_rate=anneal_rate,
            eval_temperature=eval_temperature,
            entropy_weight=entropy_weight,
            sparsity_target=sparsity_target,
            sparsity_weight=sparsity_weight,
            hc_beta=hc_beta,
            hc_gamma=hc_gamma,
            hc_zeta=hc_zeta,
            grad_scale=grad_scale,
            head_init=head_init,
            head_init_std=head_init_std,
            return_aux=return_aux,
            encoder_layers=encoder_layers,
            encoder_num_heads=encoder_num_heads,
        )

    def _split_pe(self, pe_encoding: PEType) -> tuple[tuple[torch.Tensor, ...], str]:
        if isinstance(pe_encoding, torch.Tensor):
            return (pe_encoding,), "tensor"
        if isinstance(pe_encoding, tuple):
            if len(pe_encoding) != 2:
                raise ValueError("PE tuple input must be a RoPE pair `(cos, sin)`.")
            return tuple(pe_encoding), "tuple"
        if isinstance(pe_encoding, list):
            if len(pe_encoding) != 2:
                raise ValueError("PE list input must be a RoPE pair `[cos, sin]`.")
            return tuple(pe_encoding), "list"
        raise TypeError(f"Unsupported pe_encoding type: {type(pe_encoding)}")

    def _merge_pe(self, pe_components: tuple[torch.Tensor, ...], kind: str) -> PEType:
        if kind == "tensor":
            return pe_components[0]
        if kind == "tuple":
            return tuple(pe_components)
        return list(pe_components)

    def _expand_gate(self, token_gate: torch.Tensor, valid_len: int, pe_dim: int) -> torch.Tensor:
        if token_gate.dim() == 2:
            token_gate = token_gate.unsqueeze(-1)
        if token_gate.dim() != 3:
            raise ValueError(f"Expected token gate to be [B, L, 1] or [L, 1], got {tuple(token_gate.shape)}.")

        token_gate = token_gate[:, :valid_len, :]
        if token_gate.shape[-1] == 1:
            token_gate = token_gate.expand(-1, -1, pe_dim)
        if token_gate.shape[0] == 1:
            return token_gate[0]
        return token_gate.mean(dim=0)

    def _expand_gate_batched(self, token_gate: torch.Tensor, valid_len: int, pe_dim: int) -> torch.Tensor:
        """Variant of _expand_gate that preserves the batch dim, returning [B, valid_len, pe_dim]."""
        if token_gate.dim() == 2:
            token_gate = token_gate.unsqueeze(-1)
        if token_gate.dim() != 3:
            raise ValueError(f"Expected token gate to be [B, L, 1] or [L, 1], got {tuple(token_gate.shape)}.")
        token_gate = token_gate[:, :valid_len, :]
        if token_gate.shape[-1] == 1:
            token_gate = token_gate.expand(-1, -1, pe_dim)
        return token_gate

    def _normalize_sparse_selection(
        self,
        sparse_token_selection: SparseTokenSelection | torch.Tensor,
        device: torch.device,
    ) -> SparseTokenSelection:
        if isinstance(sparse_token_selection, SparseTokenSelection):
            main_token_indices = sparse_token_selection.main_token_indices.to(device=device, dtype=torch.long)
            ref_token_indices = sparse_token_selection.ref_token_indices
            if ref_token_indices is not None:
                ref_token_indices = ref_token_indices.to(device=device, dtype=torch.long)
            return SparseTokenSelection(
                main_token_indices=main_token_indices,
                ref_token_indices=ref_token_indices,
                token_mask=sparse_token_selection.token_mask,
                token_coords=sparse_token_selection.token_coords,
                crop_token_indices=sparse_token_selection.crop_token_indices,
                crop_token_coords=sparse_token_selection.crop_token_coords,
                crop_bounds=sparse_token_selection.crop_bounds,
            )

        main_token_indices = torch.as_tensor(sparse_token_selection, device=device, dtype=torch.long).view(-1)
        ref_token_indices = torch.arange(main_token_indices.shape[0], device=device, dtype=torch.long)
        return SparseTokenSelection(
            main_token_indices=main_token_indices,
            ref_token_indices=ref_token_indices,
        )

    def _build_sparse_swapped_freqs(
        self,
        base_img_freqs: torch.Tensor,
        img_ids: torch.Tensor,
        sparse_token_selection: SparseTokenSelection | torch.Tensor,
        ref_t_coord: int = 10,
        main_t_coord: int = 0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if img_ids.dim() != 2 or img_ids.shape[-1] < 3:
            raise ValueError(f"Expected img_ids shape [seq, 4] or [seq, >=3], got {tuple(img_ids.shape)}.")
        if base_img_freqs.dim() != 2:
            raise ValueError(f"Expected base image PE to be [seq, dim], got {tuple(base_img_freqs.shape)}.")
        if base_img_freqs.shape[0] != img_ids.shape[0]:
            raise ValueError(
                f"Image PE length {base_img_freqs.shape[0]} must match img_ids length {img_ids.shape[0]}."
            )

        t_ids = img_ids[:, 0].to(torch.int64)
        h_ids = img_ids[:, 1].to(torch.int64)
        w_ids = img_ids[:, 2].to(torch.int64)

        main_mask = t_ids == int(main_t_coord)
        ref_mask = t_ids == int(ref_t_coord)
        if not torch.any(main_mask):
            raise ValueError(f"No main image tokens found for t={main_t_coord}.")
        if not torch.any(ref_mask):
            raise ValueError(f"No reference image tokens found for t={ref_t_coord}.")

        main_h = int(h_ids[main_mask].max().item()) + 1
        main_w = int(w_ids[main_mask].max().item()) + 1
        ref_h = int(h_ids[ref_mask].max().item()) + 1
        ref_w = int(w_ids[ref_mask].max().item()) + 1

        sparse_token_selection = self._normalize_sparse_selection(sparse_token_selection, device=base_img_freqs.device)
        main_local_indices = sparse_token_selection.main_token_indices
        ref_local_indices = sparse_token_selection.ref_token_indices
        if ref_local_indices is None:
            ref_local_indices = torch.arange(main_local_indices.shape[0], device=base_img_freqs.device, dtype=torch.long)

        main_positions = main_mask.nonzero(as_tuple=False).squeeze(-1)
        ref_positions = ref_mask.nonzero(as_tuple=False).squeeze(-1)

        if main_local_indices.numel() != ref_local_indices.numel():
            raise ValueError(
                f"Selected main/ref token counts must match, got {main_local_indices.numel()} and {ref_local_indices.numel()}."
            )
        if main_local_indices.numel() == 0:
            return base_img_freqs.clone(), {
                "main_hw": (main_h, main_w),
                "ref_hw": (ref_h, ref_w),
                "main_seq_len": int(main_mask.sum().item()),
                "ref_seq_len": int(ref_mask.sum().item()),
                "selected_token_count": 0,
                "ref_t_coord": int(ref_t_coord),
                "main_t_coord": int(main_t_coord),
                "sparse_token_selection": sparse_token_selection,
            }
        if int(main_local_indices.max().item()) >= main_positions.shape[0]:
            raise ValueError(
                f"main_token_indices max {int(main_local_indices.max().item())} exceeds main token count {main_positions.shape[0]}."
            )
        if int(ref_local_indices.max().item()) >= ref_positions.shape[0]:
            raise ValueError(
                f"ref_token_indices max {int(ref_local_indices.max().item())} exceeds ref token count {ref_positions.shape[0]}."
            )

        main_seq_positions = main_positions[main_local_indices]
        ref_seq_positions = ref_positions[ref_local_indices]

        swapped = base_img_freqs.clone()
        swapped[ref_seq_positions] = base_img_freqs[main_seq_positions].to(swapped.dtype)

        aux = {
            "main_hw": (main_h, main_w),
            "ref_hw": (ref_h, ref_w),
            "main_seq_len": int(main_mask.sum().item()),
            "ref_seq_len": int(ref_mask.sum().item()),
            "selected_token_count": int(main_local_indices.shape[0]),
            "ref_t_coord": int(ref_t_coord),
            "main_t_coord": int(main_t_coord),
            "sparse_token_selection": sparse_token_selection,
        }
        return swapped, aux

    def _build_sparse_swapped_freqs_batched(
        self,
        base_img_freqs: torch.Tensor,
        img_ids: torch.Tensor,
        sparse_token_selections: list,
        ref_t_coord: int = 10,
        main_t_coord: int = 0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Per-batch sparse swap; loops over the batch and stacks. Mirrors the hard model's
        batched path so the soft variant accepts the cache pipeline's list-of-selections +
        [B, L, D] PE (the 2D `_build_sparse_swapped_freqs` only handles a single sample)."""
        if img_ids.dim() != 3:
            raise ValueError(f"Batched path expects img_ids [B, L, axes], got {tuple(img_ids.shape)}.")
        batch_size = img_ids.shape[0]
        if len(sparse_token_selections) != batch_size:
            raise ValueError(
                f"sparse_token_selections length ({len(sparse_token_selections)}) "
                f"must match batch size ({batch_size})."
            )
        swapped_list = []
        selected_counts = []
        for b in range(batch_size):
            sample_base = base_img_freqs[b] if base_img_freqs.dim() == 3 else base_img_freqs
            sample_swapped, sample_aux = self._build_sparse_swapped_freqs(
                sample_base,
                img_ids[b],
                sparse_token_selection=sparse_token_selections[b],
                ref_t_coord=ref_t_coord,
                main_t_coord=main_t_coord,
            )
            swapped_list.append(sample_swapped)
            selected_counts.append(sample_aux.get("selected_token_count", 0))
        swapped = torch.stack(swapped_list, dim=0)
        aux = {
            "main_t_coord": int(main_t_coord),
            "ref_t_coord": int(ref_t_coord),
            "selected_token_counts": selected_counts,
            "batched": True,
        }
        return swapped, aux

    def forward(
        self,
        hidden_states: torch.Tensor,
        pe_encoding: PEType,
        img_ids: torch.Tensor,
        sparse_token_selection: Optional[SparseTokenSelection | torch.Tensor | list],
        layer_idx: int,
        ref_t_coord: int = 10,
        main_t_coord: int = 0,
        return_dict: bool = True,
        ste_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Union[PEExchangeOutput, PEType]:
        pe_components, pe_kind = self._split_pe(pe_encoding)
        base_img_freqs = pe_components[0]

        if sparse_token_selection is None:
            output = PEExchangeOutput(
                pe_encoding=self._merge_pe(pe_components, pe_kind),
                token_gate=None,
                swapped_pe=None,
                base_pe=base_img_freqs,
                aux={"reason": "sparse_token_selection is None"},
            )
            return output if return_dict else output.pe_encoding

        # Dispatch on whether sparse_token_selection is a per-batch list (cache pipeline
        # always passes a list + [B, L, D] PE, even at bs=1). Mirrors the hard model.
        is_batched = isinstance(sparse_token_selection, list)

        if is_batched:
            swapped_pe, swap_aux = self._build_sparse_swapped_freqs_batched(
                base_img_freqs,
                img_ids,
                sparse_token_selections=sparse_token_selection,
                ref_t_coord=ref_t_coord,
                main_t_coord=main_t_coord,
            )
            swapped_components = [swapped_pe]
            for component in pe_components[1:]:
                component_swapped, _ = self._build_sparse_swapped_freqs_batched(
                    component,
                    img_ids,
                    sparse_token_selections=sparse_token_selection,
                    ref_t_coord=ref_t_coord,
                    main_t_coord=main_t_coord,
                )
                swapped_components.append(component_swapped)
            token_gate, ste_aux = self.ste(
                hidden_states,
                layer_idx=layer_idx,
                src_key_padding_mask=ste_key_padding_mask,
            )
            expanded_gate = self._expand_gate_batched(token_gate, swapped_pe.shape[1], swapped_pe.shape[2])
            mixed_components = []
            for base_component, swapped_component in zip(pe_components, swapped_components):
                if base_component.dim() == 2:
                    base_for_mix = base_component.unsqueeze(0).expand_as(swapped_component)
                else:
                    base_for_mix = base_component
                mixed_components.append(swapped_component * (1 - expanded_gate) + base_for_mix * expanded_gate)
        else:
            if img_ids.dim() == 3:
                img_ids = img_ids[0]
            swapped_pe, swap_aux = self._build_sparse_swapped_freqs(
                base_img_freqs,
                img_ids,
                sparse_token_selection=sparse_token_selection,
                ref_t_coord=ref_t_coord,
                main_t_coord=main_t_coord,
            )
            swapped_components = [swapped_pe]
            for component in pe_components[1:]:
                component_swapped, _ = self._build_sparse_swapped_freqs(
                    component,
                    img_ids,
                    sparse_token_selection=sparse_token_selection,
                    ref_t_coord=ref_t_coord,
                    main_t_coord=main_t_coord,
                )
                swapped_components.append(component_swapped)
            token_gate, ste_aux = self.ste(
                hidden_states,
                layer_idx=layer_idx,
                src_key_padding_mask=ste_key_padding_mask,
            )
            expanded_gate = self._expand_gate(token_gate, swapped_pe.shape[0], swapped_pe.shape[1])
            mixed_components = [
                swapped_component * (1 - expanded_gate) + base_component * expanded_gate
                for base_component, swapped_component in zip(pe_components, swapped_components)
            ]

        merged_pe = self._merge_pe(tuple(mixed_components), pe_kind)
        output = PEExchangeOutput(
            pe_encoding=merged_pe,
            token_gate=expanded_gate,
            swapped_pe=swapped_pe,
            base_pe=base_img_freqs,
            aux={
                "raw_gate": token_gate,
                "ste_aux": ste_aux,
                "pe_component_count": len(pe_components),
                **swap_aux,
            },
        )
        return output if return_dict else output.pe_encoding


PackedPEExchangeModel = PackedSparsePEExchangeModel
