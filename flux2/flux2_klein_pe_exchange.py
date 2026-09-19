from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import PIL
import torch


REPO_ROOT = Path(__file__).resolve().parent
DIFFUSERS_SRC = REPO_ROOT / "diffusers" / "src"
if str(DIFFUSERS_SRC) not in sys.path:
    sys.path.insert(0, str(DIFFUSERS_SRC))

from diffusers.pipelines.flux2.pipeline_flux2_klein import (  # noqa: E402
    Flux2KleinPipeline,
    Flux2PipelineOutput,
    XLA_AVAILABLE,
    compute_empirical_mu,
    retrieve_timesteps,
)
from diffusers.models.transformers.transformer_flux2 import (  # noqa: E402
    Flux2KVCache,
    Flux2KVAttnProcessor,
    Flux2KVParallelSelfAttnProcessor,
    Flux2Transformer2DModel,
    Flux2Transformer2DModelOutput,
    _blend_double_block_mods,
    _blend_single_block_mods,
    _flux2_kv_causal_attention,
    _get_qkv_projections,
    apply_rotary_emb,
    dispatch_attention_fn,
)
from diffusers.utils import apply_lora_scale  # noqa: E402

from qwen_pe_exchange_sparse_model import PackedSparsePEExchangeModel, SparseTokenSelection  # noqa: E402


if XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm


def _blend_mod_params_at_end(img_params, ref_params, num_ref: int, seq_len: int):
    blended = []
    for im, rm in zip(img_params, ref_params):
        if im.ndim == 2:
            im = im.unsqueeze(1)
            rm = rm.unsqueeze(1)
        B = im.shape[0]
        num_main = seq_len - num_ref
        im_main = im.expand(B, seq_len, -1)[:, :num_main, :]
        rm_end = rm.expand(B, num_ref, -1)
        blended.append(torch.cat([im_main, rm_end], dim=1))
    return tuple(blended)


def _blend_double_block_mods_at_end(
    img_mod: torch.Tensor,
    ref_mod: torch.Tensor,
    num_cond: int,
    seq_len: int,
) -> torch.Tensor:
    if img_mod.ndim == 2:
        img_mod = img_mod.unsqueeze(1)
        ref_mod = ref_mod.unsqueeze(1)
    img_chunks = torch.chunk(img_mod, 6, dim=-1)
    ref_chunks = torch.chunk(ref_mod, 6, dim=-1)
    img_mods = (img_chunks[0:3], img_chunks[3:6])
    ref_mods = (ref_chunks[0:3], ref_chunks[3:6])

    all_params = []
    for img_set, ref_set in zip(img_mods, ref_mods):
        blended = _blend_mod_params_at_end(img_set, ref_set, num_cond, seq_len)
        all_params.extend(blended)
    return torch.cat(all_params, dim=-1)


def _blend_single_block_mods_at_end(
    single_mod: torch.Tensor,
    ref_mod: torch.Tensor,
    num_cond: int,
    seq_len: int,
) -> torch.Tensor:
    if single_mod.ndim == 2:
        single_mod = single_mod.unsqueeze(1)
        ref_mod = ref_mod.unsqueeze(1)
    img_params = torch.chunk(single_mod, 3, dim=-1)
    ref_params = torch.chunk(ref_mod, 3, dim=-1)

    blended = []
    for im, rm in zip(img_params, ref_params):
        B = im.shape[0]
        im_expanded = im.expand(B, seq_len, -1)
        cond_start = seq_len - num_cond
        rm_expanded = rm.expand(B, num_cond, -1)
        blended.append(torch.cat([im_expanded[:, :cond_start, :], rm_expanded], dim=1))
    return torch.cat(blended, dim=-1)


def _apply_rotary_emb_maybe_batched(
    x: torch.Tensor,
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    sequence_dim: int = 1,
) -> torch.Tensor:
    """Apply RoPE to `x` [B, S, heads, D], supporting per-sample batched cos/sin.

    diffusers' `apply_rotary_emb` only accepts 2D `(cos, sin)` of shape [S, D] shared
    across the batch. Batched PE exchange (`_cat_rotary_emb` when the image rotary is
    per-sample) yields cos/sin of shape [B, S, D]; for those we broadcast over the heads
    dim ([B, S, 1, D]) instead of delegating. 2D cos/sin keep the original diffusers path,
    so the non-batched (bs=1 / no-exchange) behavior is byte-for-byte unchanged.
    """
    cos, sin = image_rotary_emb
    if cos.ndim == 3:  # per-sample batched rotary [B, S, D]
        cos = cos[:, :, None, :]
        sin = sin[:, :, None, :]
        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return apply_rotary_emb(x, image_rotary_emb, sequence_dim=sequence_dim)


class _Flux2CrossAttentionKVAttnProcessor(Flux2KVAttnProcessor):
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        kv_cache=None,
        kv_cache_mode: str | None = None,
        num_ref_tokens: int = 0,
        cross_attention_recorder: Callable[..., None] | None = None,
        cross_attention_layer_idx: int | None = None,
        cross_attention_branch: str | None = None,
        cross_attention_img_ids: torch.Tensor | None = None,
        cross_attention_selection: Any | None = None,
        cross_attention_main_t_coord: int = 0,
        cross_attention_ref_t_coord: int = 20,
        cross_attention_num_txt_tokens: int = 0,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = _apply_rotary_emb_maybe_batched(query, image_rotary_emb, sequence_dim=1)
            key = _apply_rotary_emb_maybe_batched(key, image_rotary_emb, sequence_dim=1)

        num_txt_tokens = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
        if num_txt_tokens == 0 and cross_attention_num_txt_tokens:
            num_txt_tokens = int(cross_attention_num_txt_tokens)

        if kv_cache_mode == "extract" and kv_cache is not None and num_ref_tokens > 0:
            ref_start = num_txt_tokens
            ref_end = num_txt_tokens + num_ref_tokens
            kv_cache.store(key[:, ref_start:ref_end].clone(), value[:, ref_start:ref_end].clone())

        can_record = (
            cross_attention_recorder is not None
            and cross_attention_layer_idx is not None
            and cross_attention_branch is not None
            and cross_attention_img_ids is not None
            and cross_attention_selection is not None
            and not isinstance(cross_attention_selection, list)
        )

        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            hidden_states = _flux2_kv_causal_attention(
                query, key, value, num_txt_tokens, num_ref_tokens, backend=self._attention_backend
            )
        elif kv_cache_mode == "cached" and kv_cache is not None:
            hidden_states = _flux2_kv_causal_attention(
                query, key, value, num_txt_tokens, 0, kv_cache=kv_cache, backend=self._attention_backend
            )
        else:
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )

        if can_record:
            cross_attention_recorder(
                layer_idx=int(cross_attention_layer_idx),
                branch=str(cross_attention_branch),
                query=query,
                key=key,
                img_ids=cross_attention_img_ids,
                selection=cross_attention_selection,
                main_t_coord=int(cross_attention_main_t_coord),
                ref_t_coord=int(cross_attention_ref_t_coord),
                num_txt_tokens=int(num_txt_tokens),
            )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


class _Flux2CrossAttentionKVParallelSelfAttnProcessor(Flux2KVParallelSelfAttnProcessor):
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        kv_cache=None,
        kv_cache_mode: str | None = None,
        num_txt_tokens: int = 0,
        num_ref_tokens: int = 0,
        cross_attention_recorder: Callable[..., None] | None = None,
        cross_attention_layer_idx: int | None = None,
        cross_attention_branch: str | None = None,
        cross_attention_img_ids: torch.Tensor | None = None,
        cross_attention_selection: Any | None = None,
        cross_attention_main_t_coord: int = 0,
        cross_attention_ref_t_coord: int = 20,
        cross_attention_num_txt_tokens: int = 0,
    ) -> torch.Tensor:
        hidden_states_proj = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states_proj, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )

        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = _apply_rotary_emb_maybe_batched(query, image_rotary_emb, sequence_dim=1)
            key = _apply_rotary_emb_maybe_batched(key, image_rotary_emb, sequence_dim=1)

        if num_txt_tokens == 0 and cross_attention_num_txt_tokens:
            num_txt_tokens = int(cross_attention_num_txt_tokens)

        if kv_cache_mode == "extract" and kv_cache is not None and num_ref_tokens > 0:
            ref_start = num_txt_tokens
            ref_end = num_txt_tokens + num_ref_tokens
            kv_cache.store(key[:, ref_start:ref_end].clone(), value[:, ref_start:ref_end].clone())

        can_record = (
            cross_attention_recorder is not None
            and cross_attention_layer_idx is not None
            and cross_attention_branch is not None
            and cross_attention_img_ids is not None
            and cross_attention_selection is not None
            and not isinstance(cross_attention_selection, list)
        )

        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            attn_output = _flux2_kv_causal_attention(
                query, key, value, num_txt_tokens, num_ref_tokens, backend=self._attention_backend
            )
        elif kv_cache_mode == "cached" and kv_cache is not None:
            attn_output = _flux2_kv_causal_attention(
                query, key, value, num_txt_tokens, 0, kv_cache=kv_cache, backend=self._attention_backend
            )
        else:
            attn_output = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )

        if can_record:
            cross_attention_recorder(
                layer_idx=int(cross_attention_layer_idx),
                branch=str(cross_attention_branch),
                query=query,
                key=key,
                img_ids=cross_attention_img_ids,
                selection=cross_attention_selection,
                main_t_coord=int(cross_attention_main_t_coord),
                ref_t_coord=int(cross_attention_ref_t_coord),
                num_txt_tokens=int(num_txt_tokens),
            )

        attn_output = attn_output.flatten(2, 3)
        attn_output = attn_output.to(query.dtype)

        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)
        return hidden_states


class Flux2KleinPEExchangeTransformer2DModel(Flux2Transformer2DModel):
    def __init__(self, *args, pe_exchange_kwargs: Optional[dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)

        total_layers = len(self.transformer_blocks) + len(self.single_transformer_blocks)
        pe_exchange_kwargs = dict(pe_exchange_kwargs or {})
        pe_exchange_kwargs.setdefault("dim_in", self.inner_dim)
        pe_exchange_kwargs.setdefault("pe_dim", sum(self.config.axes_dims_rope))
        pe_exchange_kwargs.setdefault("num_layers", total_layers)
        self.pe_exchange = PackedSparsePEExchangeModel(**pe_exchange_kwargs)
        self._ste_gate_records: dict[str, dict[int, dict[str, list[torch.Tensor]]]] | None = None
        self._ste_gate_main_coords: torch.Tensor | None = None
        self._ste_gate_ref_ids: torch.Tensor | None = None
        self._ste_gate_collection_active = False
        self._ste_gate_collection_branch: str | None = None
        self._ste_gate_capture_raw = False
        self._ste_gate_capture_branches: tuple[str, ...] = ("cond",)
        self._cross_attention_records: dict[str, dict[int, dict[str, list[torch.Tensor]]]] | None = None
        self._cross_attention_layers: frozenset[int] = frozenset()
        self._cross_attention_collection_active = False
        self._cross_attention_collection_branch: str | None = None
        self._cross_attention_capture_branches: tuple[str, ...] = ("cond",)

        for block in self.transformer_blocks:
            block.attn.set_processor(_Flux2CrossAttentionKVAttnProcessor())
        for block in self.single_transformer_blocks:
            block.attn.set_processor(_Flux2CrossAttentionKVParallelSelfAttnProcessor())

    def start_ste_gate_collection(
        self,
        *,
        capture_raw: bool = False,
        branches: tuple[str, ...] = ("cond",),
    ) -> None:
        """Start a fresh per-layer STE trace.

        The legacy/default path stores only conditional hard gates. Research runs can
        additionally store logits/probabilities and trace both CFG branches.
        """
        normalized = tuple(dict.fromkeys(str(branch).strip() for branch in branches if str(branch).strip()))
        if not normalized:
            raise ValueError("STE gate collection requires at least one branch name.")
        self._ste_gate_records = {branch: {} for branch in normalized}
        self._ste_gate_main_coords = None
        self._ste_gate_ref_ids = None
        self._ste_gate_capture_raw = bool(capture_raw)
        self._ste_gate_capture_branches = normalized
        self._ste_gate_collection_branch = normalized[0]
        self._ste_gate_collection_active = True

    def pause_ste_gate_collection(self) -> None:
        """Temporarily suppress collection."""
        self._ste_gate_collection_active = False
        self._ste_gate_collection_branch = None

    def resume_ste_gate_collection(self) -> None:
        """Legacy helper: resume the conditional branch."""
        if self._ste_gate_records is not None:
            branch = "cond" if "cond" in self._ste_gate_capture_branches else self._ste_gate_capture_branches[0]
            self._ste_gate_collection_branch = branch
            self._ste_gate_collection_active = True

    def set_ste_gate_collection_branch(self, branch: str | None) -> None:
        """Select the CFG branch receiving subsequent layer records, or pause with ``None``."""
        if self._ste_gate_records is None:
            return
        if branch is None:
            self.pause_ste_gate_collection()
            return
        branch = str(branch)
        if branch not in self._ste_gate_capture_branches:
            raise ValueError(
                f"STE branch {branch!r} was not enabled; enabled branches={self._ste_gate_capture_branches}."
            )
        self._ste_gate_collection_branch = branch
        self._ste_gate_collection_active = True

    def pop_ste_gate_collection(self) -> dict[str, Any] | None:
        """Return CPU trace tensors and reset collection state."""
        records = self._ste_gate_records
        coords = self._ste_gate_main_coords
        ref_ids = self._ste_gate_ref_ids
        capture_raw = self._ste_gate_capture_raw
        self._ste_gate_records = None
        self._ste_gate_main_coords = None
        self._ste_gate_ref_ids = None
        self._ste_gate_collection_active = False
        self._ste_gate_collection_branch = None
        self._ste_gate_capture_raw = False
        self._ste_gate_capture_branches = ("cond",)
        if not records or coords is None:
            return None

        stacked_branches: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
        for branch, branch_records in records.items():
            stacked_layers: dict[int, dict[str, torch.Tensor]] = {}
            for layer_idx, metric_records in sorted(branch_records.items()):
                stacked_metrics = {
                    metric: torch.stack(metric_values, dim=0).cpu()
                    for metric, metric_values in metric_records.items()
                    if metric_values
                }
                if stacked_metrics:
                    stacked_layers[int(layer_idx)] = stacked_metrics
            if stacked_layers:
                stacked_branches[branch] = stacked_layers

        conditional = stacked_branches.get("cond", {})
        result: dict[str, Any] = {
            "branches": stacked_branches,
            "main_token_coords": coords.cpu(),
            "ref_token_ids": ref_ids.cpu() if ref_ids is not None else None,
            "capture_raw": capture_raw,
        }
        # Backward-compatible view consumed by the existing PNG exporter.
        result["raw_gates_by_layer"] = {
            layer_idx: metrics["hard_gate"]
            for layer_idx, metrics in conditional.items()
            if "hard_gate" in metrics
        }
        return result

    def set_pe_gate_overrides(self, overrides: dict[int, float] | None) -> None:
        self.pe_exchange.set_layer_gate_overrides(overrides)

    def clear_pe_gate_overrides(self) -> None:
        self.pe_exchange.clear_layer_gate_overrides()

    def start_cross_attention_collection(
        self,
        *,
        layers: tuple[int, ...] | list[int],
        branches: tuple[str, ...] = ("cond",),
    ) -> None:
        normalized_layers = frozenset(int(layer_idx) for layer_idx in layers)
        if not normalized_layers:
            raise ValueError("Cross-attention collection requires at least one layer.")
        normalized_branches = tuple(dict.fromkeys(str(branch).strip() for branch in branches if str(branch).strip()))
        if not normalized_branches:
            raise ValueError("Cross-attention collection requires at least one branch name.")
        self._cross_attention_records = {branch: {} for branch in normalized_branches}
        self._cross_attention_layers = normalized_layers
        self._cross_attention_capture_branches = normalized_branches
        self._cross_attention_collection_branch = normalized_branches[0]
        self._cross_attention_collection_active = True

    def pause_cross_attention_collection(self) -> None:
        self._cross_attention_collection_active = False
        self._cross_attention_collection_branch = None

    def resume_cross_attention_collection(self) -> None:
        if self._cross_attention_records is not None:
            branch = (
                "cond"
                if "cond" in self._cross_attention_capture_branches
                else self._cross_attention_capture_branches[0]
            )
            self._cross_attention_collection_branch = branch
            self._cross_attention_collection_active = True

    def set_cross_attention_collection_branch(self, branch: str | None) -> None:
        if self._cross_attention_records is None:
            return
        if branch is None:
            self.pause_cross_attention_collection()
            return
        branch = str(branch)
        if branch not in self._cross_attention_capture_branches:
            raise ValueError(
                f"Cross-attention branch {branch!r} was not enabled; "
                f"enabled branches={self._cross_attention_capture_branches}."
            )
        self._cross_attention_collection_branch = branch
        self._cross_attention_collection_active = True

    def pop_cross_attention_collection(self) -> dict[str, Any] | None:
        records = self._cross_attention_records
        layers = sorted(self._cross_attention_layers)
        self._cross_attention_records = None
        self._cross_attention_layers = frozenset()
        self._cross_attention_collection_active = False
        self._cross_attention_collection_branch = None
        self._cross_attention_capture_branches = ("cond",)
        if not records:
            return None

        stacked_branches: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
        for branch, branch_records in records.items():
            stacked_layers: dict[int, dict[str, torch.Tensor]] = {}
            for layer_idx, metric_records in sorted(branch_records.items()):
                stacked_metrics = {
                    metric: torch.stack(metric_values, dim=0).cpu()
                    for metric, metric_values in metric_records.items()
                    if metric_values
                }
                if stacked_metrics:
                    stacked_layers[int(layer_idx)] = stacked_metrics
            if stacked_layers:
                stacked_branches[branch] = stacked_layers
        return {"branches": stacked_branches, "layers": layers}

    @staticmethod
    def _resolve_selected_main_ref_positions(
        *,
        img_ids: torch.Tensor,
        selection: Any,
        main_t_coord: int,
        ref_t_coord: int,
        num_txt_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = img_ids[0] if img_ids.ndim == 3 else img_ids
        if ids.ndim != 2:
            raise RuntimeError(f"Expected img_ids to be 2D or 3D, got {tuple(img_ids.shape)}.")
        if not hasattr(selection, "main_token_indices"):
            raise RuntimeError("Cross-attention collection requires a sparse token selection with main_token_indices.")

        main_positions = (ids[:, 0].to(torch.int64) == int(main_t_coord)).nonzero(as_tuple=False).squeeze(-1)
        ref_positions = (ids[:, 0].to(torch.int64) == int(ref_t_coord)).nonzero(as_tuple=False).squeeze(-1)
        main_local_indices = torch.as_tensor(selection.main_token_indices, device=ids.device, dtype=torch.long).view(-1)
        ref_local_indices = getattr(selection, "ref_token_indices", None)
        if ref_local_indices is None:
            ref_local_indices = torch.arange(main_local_indices.numel(), device=ids.device, dtype=torch.long)
        else:
            ref_local_indices = torch.as_tensor(ref_local_indices, device=ids.device, dtype=torch.long).view(-1)

        if main_local_indices.numel() != ref_local_indices.numel():
            raise RuntimeError("Cross-attention collection requires matching selected main and sub token counts.")
        if main_local_indices.numel() == 0:
            return main_local_indices, ref_local_indices
        if int(main_local_indices.max().item()) >= main_positions.numel():
            raise RuntimeError("Cross-attention main-token index is outside the main image sequence.")
        if int(ref_local_indices.max().item()) >= ref_positions.numel():
            raise RuntimeError("Cross-attention sub-token index is outside the sub sequence.")

        selected_main_positions = main_positions.index_select(0, main_local_indices) + int(num_txt_tokens)
        selected_ref_positions = ref_positions.index_select(0, ref_local_indices) + int(num_txt_tokens)
        return selected_main_positions, selected_ref_positions

    @staticmethod
    def _chunked_block_logsumexp(
        query_subset: torch.Tensor,
        key_subset: torch.Tensor,
        *,
        chunk_size: int = 64,
    ) -> torch.Tensor:
        if query_subset.numel() == 0 or key_subset.numel() == 0:
            batch, query_len, num_heads, _ = query_subset.shape
            return torch.empty((batch, query_len, num_heads), device=query_subset.device, dtype=torch.float32)

        scale = query_subset.shape[-1] ** -0.5
        key_t = key_subset.permute(0, 2, 3, 1).to(torch.float32)
        outputs: list[torch.Tensor] = []
        for start in range(0, query_subset.shape[1], chunk_size):
            end = min(start + chunk_size, query_subset.shape[1])
            query_chunk = query_subset[:, start:end].to(torch.float32)
            scores = torch.einsum("bqhd,bhdk->bqhk", query_chunk, key_t) * scale
            outputs.append(torch.logsumexp(scores, dim=-1))
        return torch.cat(outputs, dim=1)

    def _record_cross_attention(
        self,
        *,
        layer_idx: int,
        branch: str,
        query: torch.Tensor,
        key: torch.Tensor,
        img_ids: torch.Tensor,
        selection: Any,
        main_t_coord: int,
        ref_t_coord: int,
        num_txt_tokens: int,
    ) -> None:
        if (
            not self._cross_attention_collection_active
            or self._cross_attention_records is None
            or branch != self._cross_attention_collection_branch
        ):
            return

        selected_main_positions, selected_ref_positions = self._resolve_selected_main_ref_positions(
            img_ids=img_ids,
            selection=selection,
            main_t_coord=main_t_coord,
            ref_t_coord=ref_t_coord,
            num_txt_tokens=num_txt_tokens,
        )
        if selected_main_positions.numel() == 0 or selected_ref_positions.numel() == 0:
            return

        main_query = query.index_select(1, selected_main_positions)
        sub_query = query.index_select(1, selected_ref_positions)
        main_key = key.index_select(1, selected_main_positions)
        sub_key = key.index_select(1, selected_ref_positions)

        main_lse = self._chunked_block_logsumexp(main_query, key)
        sub_lse = self._chunked_block_logsumexp(sub_query, key)

        main_to_sub_logsum = self._chunked_block_logsumexp(main_query, sub_key)
        sub_to_main_logsum = self._chunked_block_logsumexp(sub_query, main_key)

        main_to_sub_mass = torch.exp(main_to_sub_logsum - main_lse).clamp_(0.0, 1.0)
        sub_to_main_mass = torch.exp(sub_to_main_logsum - sub_lse).clamp_(0.0, 1.0)

        main_to_sub_mass_by_head = main_to_sub_mass.mean(dim=1).mean(dim=0).detach()
        sub_to_main_mass_by_head = sub_to_main_mass.mean(dim=1).mean(dim=0).detach()
        main_to_sub_per_key_by_head = (main_to_sub_mass / float(sub_key.shape[1])).mean(dim=1).mean(dim=0).detach()
        sub_to_main_per_key_by_head = (sub_to_main_mass / float(main_key.shape[1])).mean(dim=1).mean(dim=0).detach()

        layer_records = self._cross_attention_records.setdefault(branch, {}).setdefault(int(layer_idx), {})
        layer_records.setdefault("main_to_sub_block_mass_by_head", []).append(main_to_sub_mass_by_head)
        layer_records.setdefault("main_to_sub_per_key_mass_by_head", []).append(main_to_sub_per_key_by_head)
        layer_records.setdefault("sub_to_main_block_mass_by_head", []).append(sub_to_main_mass_by_head)
        layer_records.setdefault("sub_to_main_per_key_mass_by_head", []).append(sub_to_main_per_key_by_head)
        layer_records.setdefault("selected_main_token_count", []).append(
            torch.tensor(int(main_key.shape[1]), device=main_key.device, dtype=torch.int32)
        )
        layer_records.setdefault("selected_sub_token_count", []).append(
            torch.tensor(int(sub_key.shape[1]), device=sub_key.device, dtype=torch.int32)
        )

    @staticmethod
    def _ste_token_scalar(value: torch.Tensor, *, name: str) -> torch.Tensor:
        if value.ndim == 3:
            if value.shape[0] != 1 or value.shape[-1] != 1:
                raise RuntimeError(f"STE {name} export expects [1, seq, 1], got {tuple(value.shape)}.")
            return value[0, :, 0]
        if value.ndim == 2 and value.shape[-1] == 1:
            return value[:, 0]
        if value.ndim == 1:
            return value
        raise RuntimeError(f"Unexpected STE {name} shape: {tuple(value.shape)}.")

    def _record_ste_gate(
        self,
        *,
        pe_output,
        img_ids: torch.Tensor,
        layer_idx: int,
        ref_t_coord: int,
        main_t_coord: int,
    ) -> None:
        branch = self._ste_gate_collection_branch
        if not self._ste_gate_collection_active or self._ste_gate_records is None or branch is None:
            return

        raw_gate = pe_output.aux.get("raw_gate")
        if raw_gate is None:
            return
        gate = self._ste_token_scalar(raw_gate, name="hard_gate")

        ids = img_ids[0] if img_ids.ndim == 3 else img_ids
        if gate.shape[0] != ids.shape[0]:
            raise RuntimeError(
                f"STE gate length {gate.shape[0]} does not match image-id length {ids.shape[0]}."
            )

        selection = pe_output.aux.get("sparse_token_selection")
        if selection is None:
            return

        main_positions = (ids[:, 0].to(torch.int64) == int(main_t_coord)).nonzero(as_tuple=False).squeeze(-1)
        ref_positions = (ids[:, 0].to(torch.int64) == int(ref_t_coord)).nonzero(as_tuple=False).squeeze(-1)
        main_local_indices = torch.as_tensor(
            selection.main_token_indices,
            device=ids.device,
            dtype=torch.long,
        ).view(-1)
        ref_local_indices = selection.ref_token_indices
        if ref_local_indices is None:
            ref_local_indices = torch.arange(main_local_indices.numel(), device=ids.device, dtype=torch.long)
        else:
            ref_local_indices = torch.as_tensor(ref_local_indices, device=ids.device, dtype=torch.long).view(-1)

        if main_local_indices.numel() != ref_local_indices.numel():
            raise RuntimeError("STE gate export requires matching selected main and sub token counts.")
        if main_local_indices.numel() == 0:
            return
        if int(main_local_indices.max().item()) >= main_positions.numel():
            raise RuntimeError("STE gate export main-token index is outside the main image sequence.")
        if int(ref_local_indices.max().item()) >= ref_positions.numel():
            raise RuntimeError("STE gate export sub-token index is outside the sub sequence.")

        selected_main_positions = main_positions.index_select(0, main_local_indices)
        selected_ref_positions = ref_positions.index_select(0, ref_local_indices)
        selected_gate = (gate.index_select(0, selected_ref_positions) >= 0.5).to(torch.uint8).detach()
        selected_main_coords = ids.index_select(0, selected_main_positions)[:, 1:3].to(torch.int64).detach()
        selected_ref_ids = ids.index_select(0, selected_ref_positions).to(torch.int64).detach()
        if self._ste_gate_main_coords is None:
            self._ste_gate_main_coords = selected_main_coords
            self._ste_gate_ref_ids = selected_ref_ids
        else:
            if self._ste_gate_main_coords.shape != selected_main_coords.shape or not torch.equal(
                self._ste_gate_main_coords, selected_main_coords
            ):
                raise RuntimeError("STE gate export requires stable selected main-token coordinates.")
            if self._ste_gate_ref_ids is None or self._ste_gate_ref_ids.shape != selected_ref_ids.shape or not torch.equal(
                self._ste_gate_ref_ids, selected_ref_ids
            ):
                raise RuntimeError("STE gate export requires stable selected sub-token IDs.")

        layer_records = self._ste_gate_records.setdefault(branch, {}).setdefault(int(layer_idx), {})
        layer_records.setdefault("hard_gate", []).append(selected_gate)
        if self._ste_gate_capture_raw:
            ste_aux = pe_output.aux.get("ste_aux") or {}
            for name in ("logits", "probs"):
                value = ste_aux.get(name)
                if value is None:
                    raise RuntimeError(f"STE raw export requested, but {name!r} is missing from ste_aux.")
                scalar = self._ste_token_scalar(value, name=name)
                selected_value = scalar.index_select(0, selected_ref_positions).detach()
                layer_records.setdefault(name, []).append(selected_value)

    @staticmethod
    def _cat_rotary_emb(
        text_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_cos, text_sin = text_rotary_emb
        img_cos, img_sin = image_rotary_emb
        if img_cos.ndim == 3:
            # Batched image PE [B, L, D]; expand text PE on batch dim, concat along sequence.
            if text_cos.ndim == 2:
                text_cos = text_cos.unsqueeze(0).expand(img_cos.shape[0], -1, -1)
                text_sin = text_sin.unsqueeze(0).expand(img_sin.shape[0], -1, -1)
            return (
                torch.cat([text_cos, img_cos], dim=1),
                torch.cat([text_sin, img_sin], dim=1),
            )
        return (
            torch.cat([text_cos, img_cos], dim=0),
            torch.cat([text_sin, img_sin], dim=0),
        )

    @staticmethod
    def _has_t_coord(img_ids: torch.Tensor, t_coord: int) -> bool:
        if img_ids.numel() == 0:
            return False
        # Flatten away batch dim if present; we only need a global "does it exist" check.
        return bool(torch.any(img_ids[..., 0].to(torch.int64) == int(t_coord)))

    def _batched_pos_embed(
        self, ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Like `self.pos_embed(ids)` but accepts `[B, S, axes]` input.

        The underlying `Flux2PosEmbed.forward` calls `torch.outer(pos, freqs)` which only
        accepts 1D `pos`, so we flatten and reshape around it.
        """
        if ids.ndim == 2:
            return self.pos_embed(ids)
        if ids.ndim != 3:
            raise ValueError(f"Expected ids to be 2D or 3D, got {tuple(ids.shape)}.")
        batch_size, seq_len, num_axes = ids.shape
        flat = ids.reshape(batch_size * seq_len, num_axes)
        cos, sin = self.pos_embed(flat)
        return (
            cos.reshape(batch_size, seq_len, -1),
            sin.reshape(batch_size, seq_len, -1),
        )

    def _maybe_apply_pe_exchange(
        self,
        hidden_states: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        img_ids: torch.Tensor,
        pe_sparse_token_selection: Optional[SparseTokenSelection | torch.Tensor | list],
        layer_idx: int,
        pe_ref_t_coord: int,
        pe_main_t_coord: int,
        pe_ste_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pe_sparse_token_selection is None:
            return image_rotary_emb
        if not self._has_t_coord(img_ids, pe_ref_t_coord):
            return image_rotary_emb

        pe_output = self.pe_exchange(
            hidden_states=hidden_states,
            pe_encoding=image_rotary_emb,
            img_ids=img_ids,
            sparse_token_selection=pe_sparse_token_selection,
            layer_idx=layer_idx,
            ref_t_coord=pe_ref_t_coord,
            main_t_coord=pe_main_t_coord,
            ste_key_padding_mask=pe_ste_key_padding_mask,
            return_dict=True,
        )
        if self._ste_gate_collection_active:
            self._record_ste_gate(
                pe_output=pe_output,
                img_ids=img_ids,
                layer_idx=layer_idx,
                ref_t_coord=pe_ref_t_coord,
                main_t_coord=pe_main_t_coord,
            )
        return pe_output.pe_encoding

    @apply_lora_scale("joint_attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
        kv_cache: Flux2KVCache | None = None,
        kv_cache_mode: str | None = None,
        num_ref_tokens: int = 0,
        ref_fixed_timestep: float = 0.0,
        pe_sparse_token_selection: Optional[SparseTokenSelection | torch.Tensor | list] = None,
        pe_ref_t_coord: int = 10,
        pe_main_t_coord: int = 0,
        num_cond_tokens: int = 0,
        pe_ste_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Flux2Transformer2DModelOutput:
        num_txt_tokens = encoder_hidden_states.shape[1]

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = self.time_guidance_embed(timestep, guidance)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)

        # Condition tokens (e.g., background + subject conditioning latents appended at the
        # end of the image stream) should be modulated as if at t=0 — matching Qwen's
        # `zero_cond_t` mechanism. We blend a separate t=0 modulation onto the trailing
        # `num_cond_tokens` positions of the image stream. Mutually exclusive with the
        # KV-cache "extract" ref-token blend.
        cond_double_mod_img = None
        cond_single_mod = None
        if num_cond_tokens > 0 and not (kv_cache_mode == "extract" and num_ref_tokens > 0):
            zero_timestep = torch.zeros_like(timestep)
            cond_temb = self.time_guidance_embed(zero_timestep, guidance)
            cond_double_mod_img = self.double_stream_modulation_img(cond_temb)
            cond_single_mod = self.single_stream_modulation(cond_temb)

            num_img_tokens = hidden_states.shape[1]
            double_stream_mod_img = _blend_double_block_mods_at_end(
                double_stream_mod_img, cond_double_mod_img, num_cond_tokens, num_img_tokens
            )

        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            num_img_tokens = hidden_states.shape[1]

            kv_cache = Flux2KVCache(
                num_double_layers=len(self.transformer_blocks),
                num_single_layers=len(self.single_transformer_blocks),
            )
            kv_cache.num_ref_tokens = num_ref_tokens

            ref_timestep = torch.full_like(timestep, ref_fixed_timestep * 1000)
            ref_temb = self.time_guidance_embed(ref_timestep, guidance)

            ref_double_mod_img = self.double_stream_modulation_img(ref_temb)
            ref_single_mod = self.single_stream_modulation(ref_temb)

            double_stream_mod_img = _blend_double_block_mods(
                double_stream_mod_img, ref_double_mod_img, num_ref_tokens, num_img_tokens
            )

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # When `pe_sparse_token_selection` is a per-batch list, the swap differs per sample
        # and we need to feed the transformer batched RoPE [B, L, D]. Otherwise (single
        # selection / no exchange), keep the standard collapsed behavior.
        batched_pe = isinstance(pe_sparse_token_selection, list)

        if batched_pe:
            if img_ids.ndim != 3:
                raise ValueError(
                    "Batched pe_sparse_token_selection (list) requires img_ids of shape "
                    f"[B, L, axes], got {tuple(img_ids.shape)}."
                )
            if txt_ids.ndim == 3:
                txt_ids = txt_ids[0]
            image_rotary_emb = self._batched_pos_embed(img_ids)
            text_rotary_emb = self.pos_embed(txt_ids)
        else:
            if img_ids.ndim == 3:
                img_ids = img_ids[0]
            if txt_ids.ndim == 3:
                txt_ids = txt_ids[0]
            image_rotary_emb = self.pos_embed(img_ids)
            text_rotary_emb = self.pos_embed(txt_ids)

        if kv_cache_mode == "extract":
            base_kv_attn_kwargs = {
                **(joint_attention_kwargs or {}),
                "kv_cache": None,
                "kv_cache_mode": "extract",
                "num_ref_tokens": num_ref_tokens,
            }
        elif kv_cache_mode == "cached" and kv_cache is not None:
            base_kv_attn_kwargs = {
                **(joint_attention_kwargs or {}),
                "kv_cache": None,
                "kv_cache_mode": "cached",
                "num_ref_tokens": kv_cache.num_ref_tokens,
            }
        else:
            base_kv_attn_kwargs = dict(joint_attention_kwargs or {})

        for index_block, block in enumerate(self.transformer_blocks):
            current_kv_attn_kwargs = dict(base_kv_attn_kwargs)
            if kv_cache_mode is not None and kv_cache is not None:
                current_kv_attn_kwargs["kv_cache"] = kv_cache.get_double(index_block)

            # Always exchange from the original RoPE, not a chained/modified one.
            layer_image_rotary_emb = self._maybe_apply_pe_exchange(
                hidden_states=hidden_states,
                image_rotary_emb=image_rotary_emb,
                img_ids=img_ids,
                pe_sparse_token_selection=pe_sparse_token_selection,
                pe_ste_key_padding_mask=pe_ste_key_padding_mask,
                layer_idx=index_block,
                pe_ref_t_coord=pe_ref_t_coord,
                pe_main_t_coord=pe_main_t_coord,
            )
            concat_rotary_emb = self._cat_rotary_emb(text_rotary_emb, layer_image_rotary_emb)
            if (
                self._cross_attention_collection_active
                and self._cross_attention_records is not None
                and self._cross_attention_collection_branch is not None
                and index_block in self._cross_attention_layers
                and pe_sparse_token_selection is not None
            ):
                current_kv_attn_kwargs.update(
                    {
                        "cross_attention_recorder": self._record_cross_attention,
                        "cross_attention_layer_idx": int(index_block),
                        "cross_attention_branch": self._cross_attention_collection_branch,
                        "cross_attention_img_ids": img_ids,
                        "cross_attention_selection": pe_sparse_token_selection,
                        "cross_attention_main_t_coord": int(pe_main_t_coord),
                        "cross_attention_ref_t_coord": int(pe_ref_t_coord),
                        "cross_attention_num_txt_tokens": int(num_txt_tokens),
                    }
                )

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    double_stream_mod_img,
                    double_stream_mod_txt,
                    concat_rotary_emb,
                    current_kv_attn_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod_img=double_stream_mod_img,
                    temb_mod_txt=double_stream_mod_txt,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=current_kv_attn_kwargs,
                )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            total_single_len = hidden_states.shape[1]
            single_stream_mod = _blend_single_block_mods(
                single_stream_mod, ref_single_mod, num_txt_tokens, num_ref_tokens, total_single_len
            )
        elif cond_single_mod is not None:
            total_single_len = hidden_states.shape[1]
            single_stream_mod = _blend_single_block_mods_at_end(
                single_stream_mod, cond_single_mod, num_cond_tokens, total_single_len
            )

        if kv_cache_mode is not None:
            base_kv_attn_kwargs_single = {**base_kv_attn_kwargs, "num_txt_tokens": num_txt_tokens}
        else:
            base_kv_attn_kwargs_single = dict(base_kv_attn_kwargs)

        image_token_hidden_states = hidden_states[:, num_txt_tokens:, ...]
        double_block_count = len(self.transformer_blocks)
        for index_block, block in enumerate(self.single_transformer_blocks):
            current_kv_attn_kwargs_single = dict(base_kv_attn_kwargs_single)
            if kv_cache_mode is not None and kv_cache is not None:
                current_kv_attn_kwargs_single["kv_cache"] = kv_cache.get_single(index_block)

            layer_idx = double_block_count + index_block
            layer_image_rotary_emb = self._maybe_apply_pe_exchange(
                hidden_states=image_token_hidden_states,
                image_rotary_emb=image_rotary_emb,
                img_ids=img_ids,
                pe_sparse_token_selection=pe_sparse_token_selection,
                pe_ste_key_padding_mask=pe_ste_key_padding_mask,
                layer_idx=layer_idx,
                pe_ref_t_coord=pe_ref_t_coord,
                pe_main_t_coord=pe_main_t_coord,
            )
            concat_rotary_emb = self._cat_rotary_emb(text_rotary_emb, layer_image_rotary_emb)
            if (
                self._cross_attention_collection_active
                and self._cross_attention_records is not None
                and self._cross_attention_collection_branch is not None
                and layer_idx in self._cross_attention_layers
                and pe_sparse_token_selection is not None
            ):
                current_kv_attn_kwargs_single.update(
                    {
                        "cross_attention_recorder": self._record_cross_attention,
                        "cross_attention_layer_idx": int(layer_idx),
                        "cross_attention_branch": self._cross_attention_collection_branch,
                        "cross_attention_img_ids": img_ids,
                        "cross_attention_selection": pe_sparse_token_selection,
                        "cross_attention_main_t_coord": int(pe_main_t_coord),
                        "cross_attention_ref_t_coord": int(pe_ref_t_coord),
                        "cross_attention_num_txt_tokens": int(num_txt_tokens),
                    }
                )

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    None,
                    single_stream_mod,
                    concat_rotary_emb,
                    current_kv_attn_kwargs_single,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=None,
                    temb_mod=single_stream_mod,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=current_kv_attn_kwargs_single,
                )

            image_token_hidden_states = hidden_states[:, num_txt_tokens:, ...]

        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            hidden_states = hidden_states[:, num_txt_tokens + num_ref_tokens :, ...]
        else:
            hidden_states = hidden_states[:, num_txt_tokens:, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if kv_cache_mode == "extract":
            if not return_dict:
                return (output, kv_cache)
            return Flux2Transformer2DModelOutput(sample=output, kv_cache=kv_cache)

        if not return_dict:
            return (output,)

        return Flux2Transformer2DModelOutput(sample=output)


class Flux2KleinPEExchangePipeline(Flux2KleinPipeline):
    @torch.no_grad()
    def __call__(
        self,
        image: list[PIL.Image.Image] | PIL.Image.Image | None = None,
        prompt: str | list[str] = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: str | list[str] | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int] = (9, 18, 27),
        pe_sparse_token_selection: Optional[SparseTokenSelection | torch.Tensor] = None,
        pe_ref_t_coord: int = 10,
        pe_main_t_coord: int = 0,
        zero_cond_t: bool = True,
    ):
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            guidance_scale=guidance_scale,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        if self.do_classifier_free_guidance:
            negative_prompt = ""
            if prompt is not None and isinstance(prompt, list):
                negative_prompt = [negative_prompt] * len(prompt)
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        if image is not None and not isinstance(image, list):
            image = [image]

        condition_images = None
        if image is not None:
            for img in image:
                self.image_processor.check_image_input(img)

            condition_images = []
            for img in image:
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = img.size

                multiple_of = self.vae_scale_factor * 2
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
                condition_images.append(img)
                height = height or image_height
                width = width or image_width

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        image_latents = None
        image_latent_ids = None
        if condition_images is not None:
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=condition_images,
                batch_size=batch_size * num_images_per_prompt,
                generator=generator,
                device=device,
                dtype=self.vae.dtype,
            )

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

        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                latent_model_input = latents.to(self.transformer.dtype)
                latent_image_ids = latent_ids

                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1).to(self.transformer.dtype)
                    latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

                num_cond_tokens = int(image_latents.shape[1]) if (zero_cond_t and image_latents is not None) else 0

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                        pe_sparse_token_selection=pe_sparse_token_selection,
                        pe_ref_t_coord=pe_ref_t_coord,
                        pe_main_t_coord=pe_main_t_coord,
                        num_cond_tokens=num_cond_tokens,
                    )[0]

                noise_pred = noise_pred[:, : latents.size(1) :]

                if self.do_classifier_free_guidance:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=None,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self._attention_kwargs,
                            return_dict=False,
                            pe_sparse_token_selection=pe_sparse_token_selection,
                            pe_ref_t_coord=pe_ref_t_coord,
                            pe_main_t_coord=pe_main_t_coord,
                            num_cond_tokens=num_cond_tokens,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1) :]
                    noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

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


def build_flux2_klein_pe_exchange_pipeline(
    pretrained_model_name_or_path: str,
    pe_exchange_kwargs: Optional[dict[str, Any]] = None,
    transformer_state_dict: Optional[dict[str, torch.Tensor]] = None,
    **from_pretrained_kwargs,
) -> Flux2KleinPEExchangePipeline:
    base_pipe = Flux2KleinPipeline.from_pretrained(pretrained_model_name_or_path, **from_pretrained_kwargs)

    config = base_pipe.transformer.config
    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    valid_keys = set(inspect.signature(Flux2Transformer2DModel.__init__).parameters.keys()) - {"self"}
    init_kwargs = {key: value for key, value in config_dict.items() if key in valid_keys}

    custom_transformer = Flux2KleinPEExchangeTransformer2DModel(
        pe_exchange_kwargs=pe_exchange_kwargs,
        **init_kwargs,
    )
    custom_transformer.load_state_dict(base_pipe.transformer.state_dict(), strict=False)
    if transformer_state_dict is not None:
        custom_transformer.load_state_dict(transformer_state_dict, strict=False)

    custom_transformer.to(dtype=base_pipe.transformer.dtype)

    return Flux2KleinPEExchangePipeline(
        scheduler=base_pipe.scheduler,
        vae=base_pipe.vae,
        text_encoder=base_pipe.text_encoder,
        tokenizer=base_pipe.tokenizer,
        transformer=custom_transformer,
        is_distilled=base_pipe.config.is_distilled,
    )
