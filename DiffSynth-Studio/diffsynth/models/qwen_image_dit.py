import torch, math
import torch.nn as nn
from typing import Tuple, Optional, Union, List
from einops import rearrange
from .general_modules import TimestepEmbeddings, RMSNorm, AdaLayerNorm

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False


def qwen_image_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, attention_mask = None, enable_fp8_attention: bool = False):
    if FLASH_ATTN_3_AVAILABLE and attention_mask is None:
        if not enable_fp8_attention:
            q = rearrange(q, "b n s d -> b s n d", n=num_heads)
            k = rearrange(k, "b n s d -> b s n d", n=num_heads)
            v = rearrange(v, "b n s d -> b s n d", n=num_heads)
            x = flash_attn_interface.flash_attn_func(q, k, v)
            if isinstance(x, tuple):
                x = x[0]
            x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
        else:
            origin_dtype = q.dtype
            q_std, k_std, v_std = q.std(), k.std(), v.std()
            q, k, v = (q / q_std).to(torch.float8_e4m3fn), (k / k_std).to(torch.float8_e4m3fn), (v / v_std).to(torch.float8_e4m3fn)
            q = rearrange(q, "b n s d -> b s n d", n=num_heads)
            k = rearrange(k, "b n s d -> b s n d", n=num_heads)
            v = rearrange(v, "b n s d -> b s n d", n=num_heads)
            x = flash_attn_interface.flash_attn_func(q, k, v, softmax_scale=q_std * k_std / math.sqrt(q.size(-1)))
            if isinstance(x, tuple):
                x = x[0]
            x = x.to(origin_dtype) * v_std
            x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    else:
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


class ApproximateGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)

def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]]
):
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    return x_out.type_as(x)


class QwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat([
            self.rope_params(pos_index, self.axes_dim[0], self.theta),
            self.rope_params(pos_index, self.axes_dim[1], self.theta),
            self.rope_params(pos_index, self.axes_dim[2], self.theta),
        ], dim=1)
        self.neg_freqs = torch.cat([
            self.rope_params(neg_index, self.axes_dim[0], self.theta),
            self.rope_params(neg_index, self.axes_dim[1], self.theta),
            self.rope_params(neg_index, self.axes_dim[2], self.theta),
        ], dim=1)
        self.rope_cache = {}
        self.scale_rope = scale_rope
        
    def rope_params(self, index, dim, theta=10000):
        """
            Args:
                index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(
            index,
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim))
        )
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs


    def _expand_pos_freqs_if_needed(self, video_fhw, txt_seq_lens):
        if isinstance(video_fhw, list):
            video_fhw = tuple(max([i[j] for i in video_fhw]) for j in range(3))
        _, height, width = video_fhw
        if self.scale_rope:
            max_vid_index = max(height // 2, width // 2)
        else:
            max_vid_index = max(height, width)
        required_len = max_vid_index + max(txt_seq_lens)
        cur_max_len = self.pos_freqs.shape[0]
        if required_len <= cur_max_len:
            return

        new_max_len = math.ceil(required_len / 512) * 512
        pos_index = torch.arange(new_max_len)
        neg_index = torch.arange(new_max_len).flip(0) * -1 - 1
        self.pos_freqs = torch.cat([
            self.rope_params(pos_index, self.axes_dim[0], self.theta),
            self.rope_params(pos_index, self.axes_dim[1], self.theta),
            self.rope_params(pos_index, self.axes_dim[2], self.theta),
        ], dim=1)
        self.neg_freqs = torch.cat([
            self.rope_params(neg_index, self.axes_dim[0], self.theta),
            self.rope_params(neg_index, self.axes_dim[1], self.theta),
            self.rope_params(neg_index, self.axes_dim[2], self.theta),
        ], dim=1)
        return


    def forward(self, video_fhw, txt_seq_lens, device):
        self._expand_pos_freqs_if_needed(video_fhw, txt_seq_lens)
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"

            if rope_key not in self.rope_cache:
                seq_lens = frame * height * width
                freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
                if self.scale_rope:
                    freqs_height = torch.cat(
                        [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0
                    )
                    freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
                    freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)

                else:
                    freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

                freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
                self.rope_cache[rope_key] = freqs.clone().contiguous()
            vid_freqs.append(self.rope_cache[rope_key])

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs


    def forward_sampling(self, video_fhw, txt_seq_lens, device):
        self._expand_pos_freqs_if_needed(video_fhw, txt_seq_lens)
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"
            if idx > 0 and f"{0}_{height}_{width}" not in self.rope_cache:
                frame_0, height_0, width_0 = video_fhw[0]

                rope_key_0 = f"0_{height_0}_{width_0}"
                spatial_freqs_0 = self.rope_cache[rope_key_0].reshape(frame_0, height_0, width_0, -1)
                h_indices = torch.linspace(0, height_0 - 1, height).long()
                w_indices = torch.linspace(0, width_0 - 1, width).long()
                h_grid, w_grid = torch.meshgrid(h_indices, w_indices, indexing='ij')
                sampled_rope = spatial_freqs_0[:, h_grid, w_grid, :]

                freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
                sampled_rope[:, :, :, :freqs_frame.shape[-1]] = freqs_frame

                seq_lens = frame * height * width
                self.rope_cache[rope_key] = sampled_rope.reshape(seq_lens, -1).clone()
            if rope_key not in self.rope_cache:
                seq_lens = frame * height * width
                freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
                if self.scale_rope:
                    freqs_height = torch.cat(
                        [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0
                    )
                    freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
                    freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)

                else:
                    freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

                freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
                self.rope_cache[rope_key] = freqs.clone()
            vid_freqs.append(self.rope_cache[rope_key].contiguous())

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs


class QwenFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(dim * 4)
        self.net = nn.ModuleList([])
        self.net.append(ApproximateGELU(dim, inner_dim))
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

class QwenDoubleStreamAttention(nn.Module):
    def __init__(
        self,
        dim_a,
        dim_b,
        num_heads,
        head_dim,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(dim_a, dim_a)
        self.to_k = nn.Linear(dim_a, dim_a)
        self.to_v = nn.Linear(dim_a, dim_a)
        self.norm_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_k = RMSNorm(head_dim, eps=1e-6)

        self.add_q_proj = nn.Linear(dim_b, dim_b)
        self.add_k_proj = nn.Linear(dim_b, dim_b)
        self.add_v_proj = nn.Linear(dim_b, dim_b)
        self.norm_added_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_added_k = RMSNorm(head_dim, eps=1e-6)

        self.to_out = torch.nn.Sequential(nn.Linear(dim_a, dim_a))
        self.to_add_out = nn.Linear(dim_b, dim_b)

    def forward(
        self,
        image: torch.FloatTensor,
        text: torch.FloatTensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        enable_fp8_attention: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        img_q, img_k, img_v = self.to_q(image), self.to_k(image), self.to_v(image)
        txt_q, txt_k, txt_v = self.add_q_proj(text), self.add_k_proj(text), self.add_v_proj(text)
        seq_txt = txt_q.shape[1]

        img_q = rearrange(img_q, 'b s (h d) -> b h s d', h=self.num_heads)
        img_k = rearrange(img_k, 'b s (h d) -> b h s d', h=self.num_heads)
        img_v = rearrange(img_v, 'b s (h d) -> b h s d', h=self.num_heads)

        txt_q = rearrange(txt_q, 'b s (h d) -> b h s d', h=self.num_heads)
        txt_k = rearrange(txt_k, 'b s (h d) -> b h s d', h=self.num_heads)
        txt_v = rearrange(txt_v, 'b s (h d) -> b h s d', h=self.num_heads)

        img_q, img_k = self.norm_q(img_q), self.norm_k(img_k)
        txt_q, txt_k = self.norm_added_q(txt_q), self.norm_added_k(txt_k)
        
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_q = apply_rotary_emb_qwen(img_q, img_freqs)
            img_k = apply_rotary_emb_qwen(img_k, img_freqs)
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs)

        joint_q = torch.cat([txt_q, img_q], dim=2)
        joint_k = torch.cat([txt_k, img_k], dim=2)
        joint_v = torch.cat([txt_v, img_v], dim=2)

        joint_attn_out = qwen_image_flash_attention(joint_q, joint_k, joint_v, num_heads=joint_q.shape[1], attention_mask=attention_mask, enable_fp8_attention=enable_fp8_attention).to(joint_q.dtype)

        txt_attn_output = joint_attn_out[:, :seq_txt, :]
        img_attn_output = joint_attn_out[:, seq_txt:, :]

        img_attn_output = self.to_out(img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_attention_heads: int, 
        attention_head_dim: int, 
        eps: float = 1e-6,
    ):    
        super().__init__()
        
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim), 
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenDoubleStreamAttention(
            dim_a=dim,
            dim_b=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = QwenFeedForward(dim=dim, dim_out=dim)

        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True), 
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = QwenFeedForward(dim=dim, dim_out=dim)
    
    def _modulate(self, x, mod_params, index=None):
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if index is not None:
            # Assuming mod_params batch dim is 2*actual_batch (chunked into 2 parts)
            # So shift, scale, gate have shape [2*actual_batch, d]
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]  # each: [actual_batch, d]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]

            # index: [b, l] where b is actual batch size
            # Expand to [b, l, 1] to match feature dimension
            index_expanded = index.unsqueeze(-1)  # [b, l, 1]

            # Expand chunks to [b, 1, d] then broadcast to [b, l, d]
            shift_0_exp = shift_0.unsqueeze(1)  # [b, 1, d]
            shift_1_exp = shift_1.unsqueeze(1)  # [b, 1, d]
            scale_0_exp = scale_0.unsqueeze(1)
            scale_1_exp = scale_1.unsqueeze(1)
            gate_0_exp = gate_0.unsqueeze(1)
            gate_1_exp = gate_1.unsqueeze(1)

            # Use torch.where to select based on index
            shift_result = torch.where(index_expanded == 0, shift_0_exp, shift_1_exp)
            scale_result = torch.where(index_expanded == 0, scale_0_exp, scale_1_exp)
            gate_result = torch.where(index_expanded == 0, gate_0_exp, gate_1_exp)
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)

        return x * (1 + scale_result) + shift_result, gate_result

    def forward(
        self,
        image: torch.Tensor,  
        text: torch.Tensor,
        temb: torch.Tensor, 
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        enable_fp8_attention = False,
        modulate_index: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        img_mod_attn, img_mod_mlp = self.img_mod(temb).chunk(2, dim=-1)  # [B, 3*dim] each
        if modulate_index is not None:
            temb = torch.chunk(temb, 2, dim=0)[0]
        txt_mod_attn, txt_mod_mlp = self.txt_mod(temb).chunk(2, dim=-1)  # [B, 3*dim] each

        img_normed = self.img_norm1(image)
        img_modulated, img_gate = self._modulate(img_normed, img_mod_attn, index=modulate_index)

        txt_normed = self.txt_norm1(text)
        txt_modulated, txt_gate = self._modulate(txt_normed, txt_mod_attn)

        img_attn_out, txt_attn_out = self.attn(
            image=img_modulated,
            text=txt_modulated,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            enable_fp8_attention=enable_fp8_attention,
        )
        
        image = image + img_gate * img_attn_out
        text = text + txt_gate * txt_attn_out

        img_normed_2 = self.img_norm2(image)
        img_modulated_2, img_gate_2 = self._modulate(img_normed_2, img_mod_mlp, index=modulate_index)

        txt_normed_2 = self.txt_norm2(text)
        txt_modulated_2, txt_gate_2 = self._modulate(txt_normed_2, txt_mod_mlp)

        img_mlp_out = self.img_mlp(img_modulated_2)
        txt_mlp_out = self.txt_mlp(txt_modulated_2)

        image = image + img_gate_2 * img_mlp_out
        text = text + txt_gate_2 * txt_mlp_out

        return text, image






class STEBlock(nn.Module):
    """Transformer-encoder-style block used before每层 STE head."""

    def __init__(self, dim_in: int = 3072, dim: int = 64, num_heads: int = 8, encoder_layers: int = 1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_down(x)
        input_dim = x.dim()
        if input_dim == 2:
            x = x.unsqueeze(0)
            out = self.encoder(x).squeeze(0)
        else:
            out = self.encoder(x)
        return out



class STE(nn.Module):
    """
    按层调用的 STE：
      - 初始化时注册 num_layers 个 Linear head
      - 每次 forward 只处理一个 layer_idx 对应的 head
      - 前向严格 {0,1}；训练期用 ST 传梯度
    """
    def __init__(
        self,
        dim_in: int = 3072,
        dim_out: int = 1,
        num_layers: int = 60,
        sampler: str = "vanilla_ste",                # "gumbel" | "bernoulli" | "hard_concrete" | "vanilla_ste"
        temperature: float = 1.0,
        min_temperature: float = 0.3,
        anneal_strategy: str = "exp",           # "exp" | "linear" | "none"
        anneal_rate: float = 3e-5,
        eval_temperature: Optional[float] = None,
        # 正则(可选)
        entropy_weight: float = 0.0,
        sparsity_target: Optional[float] = None,
        sparsity_weight: float = 0.0,
        # hard-concrete
        hc_beta: float = 2/3,
        hc_gamma: float = -0.1,
        hc_zeta: float = 1.1,
        # 其他
        grad_scale: float = 1.0,
        head_init: str = "zero",                # "zero" | "normal"
        head_init_std: float = 1e-3,
        return_aux: bool = False,
        reduce_batch: bool = True,
        encoder_layers: int = 1,
        encoder_num_heads: int = 8,
    ):
        super().__init__()
        assert sampler in ("gumbel", "bernoulli", "hard_concrete", "vanilla_ste")
        self.dim_in, self.dim_out, self.num_layers = dim_in, dim_out, num_layers
        self.sampler = sampler
        self.return_aux = return_aux
        self.reduce_batch = bool(reduce_batch)

        # 温度
        self.register_buffer("tau", torch.tensor(float(temperature)))
        self.tau_min = float(min_temperature)
        self.anneal_strategy = anneal_strategy
        self.anneal_rate = float(anneal_rate)
        self.eval_tau = float(eval_temperature) if eval_temperature is not None else None

        # 正则
        self.entropy_weight = float(entropy_weight)
        self.sparsity_target = sparsity_target
        self.sparsity_weight = float(sparsity_weight)

        # hard-concrete 参数
        self.hc_beta = float(hc_beta)
        self.hc_gamma = float(hc_gamma)
        self.hc_zeta = float(hc_zeta)

        self.grad_scale = float(grad_scale)

        # 60 个 head（每层一个）
        self.heads = nn.ModuleList([nn.Linear(64, dim_out) for _ in range(num_layers)])
        if head_init == "zero":
            for head in self.heads:
                nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        else:
            for head in self.heads:
                nn.init.normal_(head.weight, mean=0.0, std=head_init_std)
                nn.init.zeros_(head.bias)
        self.ste_blocks = nn.ModuleList([
            STEBlock(dim_in=dim_in, dim=64, num_heads=encoder_num_heads, encoder_layers=encoder_layers)
            for _ in range(num_layers)
        ])

    # --------- 工具 ---------
    @torch.no_grad()
    def _sample_logistic_noise(self, shape, device, dtype, eps: float = 1e-6):
        u = torch.rand(shape, device=device, dtype=dtype).clamp_(eps, 1.0 - eps)
        return torch.log(u) - torch.log(1.0 - u)

    def _straight_through(self, y_soft: torch.Tensor, y_hard: torch.Tensor) -> torch.Tensor:
        return y_hard + self.grad_scale * (y_soft - y_soft.detach())

    # --------- 采样器 ---------
    def _gumbel_hard(self, logits: torch.Tensor, tau: torch.Tensor):
        g = self._sample_logistic_noise(logits.shape, logits.device, logits.dtype)
        y_soft = torch.sigmoid((logits + g) / tau)
        y_hard = (y_soft >= 0.5).to(logits.dtype)
        return self._straight_through(y_soft, y_hard), y_soft

    def _bernoulli_hard(self, logits: torch.Tensor, tau: torch.Tensor):
        probs = torch.sigmoid(logits / tau)
        y_sample = torch.bernoulli(probs)
        y = self._straight_through(probs, y_sample)
        return y, probs

    def _hard_concrete(self, logits: torch.Tensor, tau: torch.Tensor):
        u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
        s = torch.sigmoid((logits + torch.log(u) - torch.log(1 - u)) / self.hc_beta)
        z_tilde = s * (self.hc_zeta - self.hc_gamma) + self.hc_gamma
        z = z_tilde.clamp_(0.0, 1.0)
        y_hard = (z >= 0.5).to(logits.dtype)
        y = self._straight_through(z, y_hard)
        probs = torch.sigmoid(logits / tau)  # 作为监控/正则的无噪声代理
        return y, probs

    def _vanilla_ste(self, logits: torch.Tensor):
        y_hard = (logits >= 0).to(logits.dtype)      # 前向严格 {0,1}
        y_soft = torch.sigmoid(logits)               # 反向代理
        y = self._straight_through(y_soft, y_hard)
        return y, y_soft

    # --------- 温度退火（建议每个 global step 调一次） ---------
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

    # --------- 前向：单层调用 ---------
    def forward(
        self,
        h: torch.Tensor,         # [B, L, dim_in] 或 [L, dim_in]
        layer_idx: int,          # 使用第 layer_idx 个 head
    ):

        ste_block = self.ste_blocks[layer_idx]
        h_enc = ste_block(h)

        head = self.heads[layer_idx]
        logits = head(h_enc)  # [L, dim_out] or [B, L, dim_out]
        tau = self.tau if self.training or self.eval_tau is None else torch.tensor(self.eval_tau, device=h.device, dtype=h.dtype)

        if self.training:
            if self.sampler == "gumbel":
                mask, probs = self._gumbel_hard(logits, tau)
            elif self.sampler == "bernoulli":
                mask, probs = self._bernoulli_hard(logits, tau)
            elif self.sampler == "hard_concrete":
                mask, probs = self._hard_concrete(logits, tau)
            else:
                mask, probs = self._vanilla_ste(logits)
        else:
            probs = torch.sigmoid(logits / tau)
            mask  = (probs >= 0.5).to(logits.dtype)

        # 可选正则（仅用本层概率）
        reg_loss = torch.as_tensor(0.0, device=h.device, dtype=h.dtype)
        if self.training and (self.entropy_weight > 0.0 or (self.sparsity_target is not None and self.sparsity_weight > 0.0)):
            p = probs.clamp(1e-6, 1 - 1e-6)
            if self.entropy_weight > 0.0:
                ent = -(p * p.log() + (1 - p) * (1 - p).log())
                reg_loss = reg_loss + ent.mean() * self.entropy_weight
            if self.sparsity_target is not None and self.sparsity_weight > 0.0:
                mean_p = p.mean()
                target = torch.tensor(self.sparsity_target, device=p.device, dtype=p.dtype)
                reg_loss = reg_loss + F.mse_loss(mean_p, target) * self.sparsity_weight

        aux = {"probs": probs, "logits": logits, "reg_loss": reg_loss, "temperature": torch.as_tensor(float(tau), device=h.device, dtype=h.dtype)}
        return mask, (aux if self.return_aux else {})

class QwenImageDiT(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 60,
    ):
        super().__init__()

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=[16,56,56], scale_rope=True) 

        self.time_text_embed = TimestepEmbeddings(256, 3072, diffusers_compatible_format=True, scale=1000, align_dtype_to_timestep=True)
        self.txt_norm = RMSNorm(3584, eps=1e-6)

        self.img_in = nn.Linear(64, 3072)
        self.txt_in = nn.Linear(3584, 3072)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=3072,
                    num_attention_heads=24,
                    attention_head_dim=128,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = AdaLayerNorm(3072, single=True)
        self.proj_out = nn.Linear(3072, 64)


    def process_entity_masks(self, latents, prompt_emb, prompt_emb_mask, entity_prompt_emb, entity_prompt_emb_mask, entity_masks, height, width, image, img_shapes):
        # prompt_emb
        all_prompt_emb = entity_prompt_emb + [prompt_emb]
        all_prompt_emb = [self.txt_in(self.txt_norm(local_prompt_emb)) for local_prompt_emb in all_prompt_emb]
        all_prompt_emb = torch.cat(all_prompt_emb, dim=1)

        # image_rotary_emb
        txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=latents.device)
        entity_seq_lens = [emb_mask.sum(dim=1).tolist() for emb_mask in entity_prompt_emb_mask]
        entity_rotary_emb = [self.pos_embed(img_shapes, entity_seq_len, device=latents.device)[1] for entity_seq_len in entity_seq_lens]
        txt_rotary_emb = torch.cat(entity_rotary_emb + [image_rotary_emb[1]], dim=0)
        image_rotary_emb = (image_rotary_emb[0], txt_rotary_emb)

        # attention_mask
        repeat_dim = latents.shape[1]
        max_masks = entity_masks.shape[1]
        entity_masks = entity_masks.repeat(1, 1, repeat_dim, 1, 1)
        entity_masks = [entity_masks[:, i, None].squeeze(1) for i in range(max_masks)]
        global_mask = torch.ones_like(entity_masks[0]).to(device=latents.device, dtype=latents.dtype)
        entity_masks = entity_masks + [global_mask]

        N = len(entity_masks)
        batch_size = entity_masks[0].shape[0]
        seq_lens = [mask_.sum(dim=1).item() for mask_ in entity_prompt_emb_mask] + [prompt_emb_mask.sum(dim=1).item()]
        total_seq_len = sum(seq_lens) + image.shape[1]
        patched_masks = []
        for i in range(N):
            patched_mask = rearrange(entity_masks[i], "B C (H P) (W Q) -> B (H W) (C P Q)", H=height//16, W=width//16, P=2, Q=2)
            patched_masks.append(patched_mask)
        attention_mask = torch.ones((batch_size, total_seq_len, total_seq_len), dtype=torch.bool).to(device=entity_masks[0].device)

        # prompt-image attention mask
        image_start = sum(seq_lens)
        image_end = total_seq_len
        cumsum = [0]
        single_image_seq = image_end - image_start
        for length in seq_lens:
            cumsum.append(cumsum[-1] + length)
        for i in range(N):
            prompt_start = cumsum[i]
            prompt_end = cumsum[i+1]
            image_mask = torch.sum(patched_masks[i], dim=-1) > 0
            image_mask = image_mask.unsqueeze(1).repeat(1, seq_lens[i], 1)
            # repeat image mask to match the single image sequence length
            repeat_time = single_image_seq // image_mask.shape[-1]
            image_mask = image_mask.repeat(1, 1, repeat_time)
            # prompt update with image
            attention_mask[:, prompt_start:prompt_end, image_start:image_end] = image_mask
            # image update with prompt
            attention_mask[:, image_start:image_end, prompt_start:prompt_end] = image_mask.transpose(1, 2)
        # prompt-prompt attention mask, let the prompt tokens not attend to each other
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                start_i, end_i = cumsum[i], cumsum[i+1]
                start_j, end_j = cumsum[j], cumsum[j+1]
                attention_mask[:, start_i:end_i, start_j:end_j] = False

        attention_mask = attention_mask.float()
        attention_mask[attention_mask == 0] = float('-inf')
        attention_mask[attention_mask == 1] = 0
        attention_mask = attention_mask.to(device=latents.device, dtype=latents.dtype).unsqueeze(1)

        return all_prompt_emb, image_rotary_emb, attention_mask


    def forward(
        self,
        latents=None,
        timestep=None,
        prompt_emb=None,
        prompt_emb_mask=None,
        height=None,
        width=None,
    ):
        img_shapes = [(latents.shape[0], latents.shape[2]//2, latents.shape[3]//2)]
        txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
        
        image = rearrange(latents, "B C (H P) (W Q) -> B (H W) (C P Q)", H=height//16, W=width//16, P=2, Q=2)
        image = self.img_in(image)
        text = self.txt_in(self.txt_norm(prompt_emb))

        conditioning = self.time_text_embed(timestep, image.dtype)

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=latents.device)

        for block in self.transformer_blocks:
            text, image = block(
                image=image,
                text=text,
                temb=conditioning,
                image_rotary_emb=image_rotary_emb,
            )
        
        image = self.norm_out(image, conditioning)
        image = self.proj_out(image)
        
        latents = rearrange(image, "B (H W) (C P Q) -> B C (H P) (W Q)", H=height//16, W=width//16, P=2, Q=2)
        return image
