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
    Flux2Transformer2DModel,
    Flux2Transformer2DModelOutput,
    _blend_double_block_mods,
    _blend_single_block_mods,
)
from diffusers.utils import apply_lora_scale  # noqa: E402

from qwen_pe_exchange_sparse_soft_model import PackedSparsePEExchangeModel, SparseTokenSelection  # noqa: E402


if XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm


class Flux2KleinPESoftExchangeTransformer2DModel(Flux2Transformer2DModel):
    def __init__(self, *args, pe_exchange_kwargs: Optional[dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)

        total_layers = len(self.transformer_blocks) + len(self.single_transformer_blocks)
        pe_exchange_kwargs = dict(pe_exchange_kwargs or {})
        pe_exchange_kwargs.setdefault("dim_in", self.inner_dim)
        pe_exchange_kwargs.setdefault("pe_dim", sum(self.config.axes_dims_rope))
        pe_exchange_kwargs.setdefault("num_layers", total_layers)
        self.pe_exchange = PackedSparsePEExchangeModel(**pe_exchange_kwargs)

    @staticmethod
    def _cat_rotary_emb(
        text_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_cos, text_sin = text_rotary_emb
        img_cos, img_sin = image_rotary_emb
        if img_cos.ndim == 3:
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
        return bool(torch.any(img_ids[..., 0].to(torch.int64) == int(t_coord)))

    def _batched_pos_embed(
        self, ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            kv_attn_kwargs = {
                **(joint_attention_kwargs or {}),
                "kv_cache": None,
                "kv_cache_mode": "extract",
                "num_ref_tokens": num_ref_tokens,
            }
        elif kv_cache_mode == "cached" and kv_cache is not None:
            kv_attn_kwargs = {
                **(joint_attention_kwargs or {}),
                "kv_cache": None,
                "kv_cache_mode": "cached",
                "num_ref_tokens": kv_cache.num_ref_tokens,
            }
        else:
            kv_attn_kwargs = joint_attention_kwargs

        for index_block, block in enumerate(self.transformer_blocks):
            if kv_cache_mode is not None and kv_cache is not None:
                kv_attn_kwargs["kv_cache"] = kv_cache.get_double(index_block)

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

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    double_stream_mod_img,
                    double_stream_mod_txt,
                    concat_rotary_emb,
                    kv_attn_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod_img=double_stream_mod_img,
                    temb_mod_txt=double_stream_mod_txt,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=kv_attn_kwargs,
                )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            total_single_len = hidden_states.shape[1]
            single_stream_mod = _blend_single_block_mods(
                single_stream_mod, ref_single_mod, num_txt_tokens, num_ref_tokens, total_single_len
            )

        if kv_cache_mode is not None:
            kv_attn_kwargs_single = {**kv_attn_kwargs, "num_txt_tokens": num_txt_tokens}
        else:
            kv_attn_kwargs_single = kv_attn_kwargs

        image_token_hidden_states = hidden_states[:, num_txt_tokens:, ...]
        double_block_count = len(self.transformer_blocks)
        for index_block, block in enumerate(self.single_transformer_blocks):
            if kv_cache_mode is not None and kv_cache is not None:
                kv_attn_kwargs_single["kv_cache"] = kv_cache.get_single(index_block)

            layer_image_rotary_emb = self._maybe_apply_pe_exchange(
                hidden_states=image_token_hidden_states,
                image_rotary_emb=image_rotary_emb,
                img_ids=img_ids,
                pe_sparse_token_selection=pe_sparse_token_selection,
                pe_ste_key_padding_mask=pe_ste_key_padding_mask,
                layer_idx=double_block_count + index_block,
                pe_ref_t_coord=pe_ref_t_coord,
                pe_main_t_coord=pe_main_t_coord,
            )
            concat_rotary_emb = self._cat_rotary_emb(text_rotary_emb, layer_image_rotary_emb)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    None,
                    single_stream_mod,
                    concat_rotary_emb,
                    kv_attn_kwargs_single,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=None,
                    temb_mod=single_stream_mod,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=kv_attn_kwargs_single,
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


class Flux2KleinPESoftExchangePipeline(Flux2KleinPipeline):
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


def build_flux2_klein_pe_soft_exchange_pipeline(
    pretrained_model_name_or_path: str,
    pe_exchange_kwargs: Optional[dict[str, Any]] = None,
    transformer_state_dict: Optional[dict[str, torch.Tensor]] = None,
    **from_pretrained_kwargs,
) -> Flux2KleinPESoftExchangePipeline:
    base_pipe = Flux2KleinPipeline.from_pretrained(pretrained_model_name_or_path, **from_pretrained_kwargs)

    config = base_pipe.transformer.config
    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    valid_keys = set(inspect.signature(Flux2Transformer2DModel.__init__).parameters.keys()) - {"self"}
    init_kwargs = {key: value for key, value in config_dict.items() if key in valid_keys}

    custom_transformer = Flux2KleinPESoftExchangeTransformer2DModel(
        pe_exchange_kwargs=pe_exchange_kwargs,
        **init_kwargs,
    )
    custom_transformer.load_state_dict(base_pipe.transformer.state_dict(), strict=False)
    if transformer_state_dict is not None:
        custom_transformer.load_state_dict(transformer_state_dict, strict=False)

    custom_transformer.to(dtype=base_pipe.transformer.dtype)

    return Flux2KleinPESoftExchangePipeline(
        scheduler=base_pipe.scheduler,
        vae=base_pipe.vae,
        text_encoder=base_pipe.text_encoder,
        tokenizer=base_pipe.tokenizer,
        transformer=custom_transformer,
        is_distilled=base_pipe.config.is_distilled,
    )
