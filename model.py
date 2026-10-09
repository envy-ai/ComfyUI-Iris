"""Iris inference, adapted from speridlabs/iris-3b a8d15239 (Apache-2.0)."""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

import comfy.ops
import comfy.model_prefetch
import comfy.model_management
import comfy.quant_ops
from comfy.ldm.common_dit import pad_to_patch_size
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.hidream.model import FeedForwardSwiGLU
from comfy.ldm.modules.attention import optimized_attention, optimized_attention_for_device


@dataclass
class IrisConfig:
    hidden_size: int = 2560
    depth: int = 24
    dual_depth: int = 8
    num_heads: int = 20
    num_kv_heads: int = 5
    patch_size: int = 16
    in_channels: int = 3
    mlp_ratio: float = 4.0
    norm_eps: float = 1e-6
    text_dim: int = 2560
    text_len: int = 300
    text_lap_num_layers: int = 12
    text_lap_num_heads: int = 32
    text_lap_mlp_ratio: float = 1.3
    timestep_max_period: float = 10.0
    rope_theta: float = 10000.0
    rope_scale: float = 16.0
    rope_aspect: str = "isotropic"
    rope_frame_pairs: int = 0
    text_rope_theta: float = 10000.0
    pixel_hidden_size: int = 16
    pixel_attn_hidden_size: int = 1280
    pixel_num_heads: int = 10
    pixel_depth: int = 4
    pixel_mlp_ratio: float = 4.0


def rotation(angles):
    cos, sin = angles.cos(), angles.sin()
    return torch.stack((cos, -sin, sin, cos), dim=-1).reshape(1, 1, angles.shape[0], angles.shape[1], 2, 2)


def image_rope(head_dim, grid, cfg, device):
    height, width = grid
    pairs = head_dim // 4
    xy_pairs = pairs - cfg.rope_frame_pairs
    freqs = cfg.rope_theta ** (-torch.arange(0, head_dim, 4, device=device, dtype=torch.float32)[:xy_pairs] / head_dim)
    if cfg.rope_aspect == "isotropic":
        step = cfg.rope_scale / max(max(height, width) - 1, 1)
        xp = torch.arange(width, device=device, dtype=torch.float32) * step
        yp = torch.arange(height, device=device, dtype=torch.float32) * step
    else:
        xp = torch.linspace(0, cfg.rope_scale, width, device=device)
        yp = torch.linspace(0, cfg.rope_scale, height, device=device)
    ax = torch.outer(xp, freqs)[None].expand(height, -1, -1)
    ay = torch.outer(yp, freqs)[:, None].expand(-1, width, -1)
    angles = torch.stack((ax, ay), dim=-1).reshape(height * width, -1)
    if cfg.rope_frame_pairs:
        angles = F.pad(angles, (0, 2 * cfg.rope_frame_pairs))
    return rotation(angles)


def text_rope(head_dim, length, theta, device):
    freqs = theta ** (-torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    return rotation(torch.outer(torch.arange(length, device=device, dtype=torch.float32), freqs))


def pixel_position(dim, height, width, device, dtype):
    # MAE axis order, with sin/cos halves for each axis.
    omega = 10000 ** (-torch.arange(dim // 4, device=device, dtype=torch.float32) / (dim / 4))
    x = torch.outer(torch.arange(width, device=device, dtype=torch.float32), omega)
    y = torch.outer(torch.arange(height, device=device, dtype=torch.float32), omega)
    x = torch.cat((x.sin(), x.cos()), dim=-1)[None].expand(height, -1, -1)
    y = torch.cat((y.sin(), y.cos()), dim=-1)[:, None].expand(-1, width, -1)
    return torch.cat((x, y), dim=-1).to(dtype)


def modulate(x, shift, scale):
    return torch.addcmul(shift, x, 1 + scale)


def attention(q, k, v, mask=None, transformer_options=None):
    selected = optimized_attention_for_device(q.device, mask=True) if mask is not None else optimized_attention
    return selected(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), q.shape[2], skip_reshape=True, mask=mask, enable_gqa=q.shape[2] != k.shape[2], transformer_options=transformer_options or {})


def norm_rope(q, k, q_norm, k_norm, rope):
    if comfy.model_management.in_training:
        q, k = apply_rope(q_norm(q).transpose(1, 2), k_norm(k).transpose(1, 2), rope)
        return q.transpose(1, 2), k.transpose(1, 2)
    with comfy.ops.CastBiasWeightContext(q_norm, q, offloadable=True) as (q_scale, _):
        with comfy.ops.CastBiasWeightContext(k_norm, k, offloadable=True) as (k_scale, _):
            return comfy.quant_ops.ck.rms_rope(q, k, rope.transpose(1, 2), q_scale, k_scale, q_norm.eps)


class SwiGLU(FeedForwardSwiGLU):
    def __init__(self, dim, ratio, operations, **kw):
        super().__init__(dim, int(dim * ratio), multiple_of=1, operations=operations, **kw)

    def forward(self, x):
        return comfy.ops.linear_input_act(self.w2, torch.cat((self.w1(x), self.w3(x)), dim=-1), "swiglu")


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, eps, operations, qk_norm=True, **kw):
        super().__init__()
        self.num_heads = heads
        self.qk_norm = qk_norm
        self.head_dim = dim // heads
        self.qkv = operations.Linear(dim, 3 * dim, bias=False, **kw)
        self.q_norm = operations.RMSNorm(self.head_dim, eps=eps, **kw) if qk_norm else nn.Identity()
        self.k_norm = operations.RMSNorm(self.head_dim, eps=eps, **kw) if qk_norm else nn.Identity()
        self.proj = operations.Linear(dim, dim, **kw)

    def forward(self, x, rope=None, mask=None, transformer_options=None):
        batch, tokens, dim = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim).unbind(2)
        if rope is not None and self.qk_norm:
            q, k = norm_rope(q, k, self.q_norm, self.k_norm, rope)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            if rope is not None:
                q, k = apply_rope(q.transpose(1, 2), k.transpose(1, 2), rope)
                q, k = q.transpose(1, 2), k.transpose(1, 2)
        return self.proj(attention(q, k, v, mask, transformer_options))


class AdapterBlock(nn.Module):
    def __init__(self, dim, heads, ratio, eps, operations, layerwise=False, **kw):
        super().__init__()
        self.norm1 = operations.RMSNorm(dim, eps=eps, **kw)
        self.attn = SelfAttention(dim, heads, eps, operations, qk_norm=not layerwise, **kw)
        self.norm2 = operations.RMSNorm(dim, eps=eps, **kw)
        if layerwise:
            hidden = int(dim * ratio)
            self.mlp = nn.Sequential(operations.Linear(dim, hidden, **kw), nn.SiLU(), operations.Linear(hidden, dim, **kw))
        else:
            self.mlp = SwiGLU(dim, ratio, operations, **kw)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)
        return x + self.mlp(self.norm2(x))


class TextRefiner(nn.Module):
    def __init__(self, cfg, operations, **kw):
        super().__init__()
        self.proj = operations.Linear(cfg.text_dim, cfg.hidden_size, **kw)
        self.blocks = nn.ModuleList([AdapterBlock(cfg.hidden_size, cfg.num_heads, cfg.mlp_ratio, cfg.norm_eps, operations, **kw) for _ in range(2)])
        self.norm = operations.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps, **kw)

    def forward(self, x, mask):
        # Every padding query retains its diagonal, matching upstream.
        keep = mask.bool()[:, None, None, :]
        keep = keep | torch.eye(x.shape[1], device=x.device, dtype=torch.bool)
        keep = torch.zeros(keep.shape, device=x.device, dtype=x.dtype).masked_fill_(~keep, -float("inf"))
        x = self.proj(x)
        for block in self.blocks:
            x = block(x, keep)
        return self.norm(x)


class LayerwiseTextEmbedder(nn.Module):
    def __init__(self, cfg, operations, **kw):
        super().__init__()
        self.layer_blocks = nn.ModuleList([AdapterBlock(cfg.text_dim, cfg.text_lap_num_heads, cfg.text_lap_mlp_ratio, cfg.norm_eps, operations, layerwise=True, **kw) for _ in range(2)])
        self.layer_pool = operations.Linear(cfg.text_lap_num_layers, 1, **kw)
        self.refiner = TextRefiner(cfg, operations, **kw)

    def forward(self, x, mask):
        batch, tokens, layers, dim = x.shape
        x = x.reshape(batch * tokens, layers, dim)
        for block in self.layer_blocks:
            x = block(x)
        x = self.layer_pool(x.transpose(1, 2)).squeeze(-1).reshape(batch, tokens, dim)
        return self.refiner(x, mask)


class SharedBias(nn.Module):
    def __init__(self, dim, device=None, dtype=None):
        super().__init__()
        self.bias = nn.Parameter(torch.empty(6 * dim, device=device, dtype=dtype))

    def forward(self, modulation):
        return modulation + comfy.ops.cast_to_input(self.bias, modulation)


class JointAttention(nn.Module):
    def __init__(self, cfg, operations, **kw):
        super().__init__()
        dim = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = dim // cfg.num_heads
        for stream in ("x", "y"):
            for projection, heads in (("q", self.num_heads), ("k", self.num_kv_heads), ("v", self.num_kv_heads)):
                setattr(self, f"{projection}_proj_{stream}", operations.Linear(dim, heads * self.head_dim, bias=False, **kw))
            setattr(self, f"q_norm_{stream}", operations.RMSNorm(self.head_dim, eps=cfg.norm_eps, **kw))
            setattr(self, f"k_norm_{stream}", operations.RMSNorm(self.head_dim, eps=cfg.norm_eps, **kw))
            setattr(self, f"proj_{stream}", operations.Linear(dim, dim, **kw))

    def project(self, x, stream, rope):
        batch, tokens, _ = x.shape
        q = getattr(self, f"q_proj_{stream}")(x).reshape(batch, tokens, self.num_heads, self.head_dim)
        k = getattr(self, f"k_proj_{stream}")(x).reshape(batch, tokens, self.num_kv_heads, self.head_dim)
        v = getattr(self, f"v_proj_{stream}")(x).reshape(batch, tokens, self.num_kv_heads, self.head_dim)
        q, k = norm_rope(q, k, getattr(self, f"q_norm_{stream}"), getattr(self, f"k_norm_{stream}"), rope)
        return q, k, v

    def forward(self, x, y, rope_img, rope_txt, gate_x, gate_y, transformer_options):
        qx, kx, vx = self.project(x, "x", rope_img)
        qy, ky, vy = self.project(y, "y", rope_txt)
        out = attention(torch.cat((qy, qx), 1), torch.cat((ky, kx), 1), torch.cat((vy, vx), 1), transformer_options=transformer_options)
        return self.proj_x(out[:, y.shape[1]:] * gate_x), self.proj_y(out[:, :y.shape[1]] * gate_y)


class DualBlock(nn.Module):
    def __init__(self, cfg, operations, **kw):
        super().__init__()
        dim, eps = cfg.hidden_size, cfg.norm_eps
        for stream in ("x", "y"):
            for number in (1, 2):
                setattr(self, f"norm_{stream}{number}", operations.RMSNorm(dim, eps=eps, **kw))
            setattr(self, f"attn_gate_{stream}", operations.Linear(dim, dim, bias=False, **kw))
            setattr(self, f"attn_post_norm_{stream}", operations.RMSNorm(dim, eps=eps, **kw))
            setattr(self, f"mlp_{stream}", SwiGLU(dim, cfg.mlp_ratio, operations, **kw))
            setattr(self, f"mlp_post_norm_{stream}", operations.RMSNorm(dim, eps=eps, **kw))
        self.attn = JointAttention(cfg, operations, **kw)
        self.adaln_img = SharedBias(dim, **kw)
        self.adaln_txt = SharedBias(dim, **kw)

    def forward(self, x, y, img_mod, txt_mod, rope_img, rope_txt, transformer_options):
        xs1, xc1, xg1, xs2, xc2, xg2 = self.adaln_img(img_mod).chunk(6, -1)
        ys1, yc1, yg1, ys2, yc2, yg2 = self.adaln_txt(txt_mod).chunk(6, -1)
        hx, hy = modulate(self.norm_x1(x), xs1, xc1), modulate(self.norm_y1(y), ys1, yc1)
        ax, ay = self.attn(hx, hy, rope_img, rope_txt, self.attn_gate_x(hx).sigmoid(), self.attn_gate_y(hy).sigmoid(), transformer_options)
        x = torch.addcmul(x, xg1, self.attn_post_norm_x(ax))
        y = torch.addcmul(y, yg1, self.attn_post_norm_y(ay))
        x = torch.addcmul(x, xg2, self.mlp_post_norm_x(self.mlp_x(modulate(self.norm_x2(x), xs2, xc2))))
        y = torch.addcmul(y, yg2, self.mlp_post_norm_y(self.mlp_y(modulate(self.norm_y2(y), ys2, yc2))))
        return x, y


class SingleBlock(nn.Module):
    def __init__(self, cfg, operations, **kw):
        super().__init__()
        dim, eps = cfg.hidden_size, cfg.norm_eps
        self.num_heads, self.num_kv_heads = cfg.num_heads, cfg.num_kv_heads
        self.head_dim = dim // cfg.num_heads
        self.norm1 = operations.RMSNorm(dim, eps=eps, **kw)
        self.norm2 = operations.RMSNorm(dim, eps=eps, **kw)
        self.q_proj = operations.Linear(dim, dim, bias=False, **kw)
        self.k_proj = operations.Linear(dim, cfg.num_kv_heads * self.head_dim, bias=False, **kw)
        self.v_proj = operations.Linear(dim, cfg.num_kv_heads * self.head_dim, bias=False, **kw)
        self.q_norm = operations.RMSNorm(self.head_dim, eps=eps, **kw)
        self.k_norm = operations.RMSNorm(self.head_dim, eps=eps, **kw)
        self.attn_gate = operations.Linear(dim, dim, bias=False, **kw)
        self.attn_proj = operations.Linear(dim, dim, **kw)
        self.attn_post_norm = operations.RMSNorm(dim, eps=eps, **kw)
        self.mlp = SwiGLU(dim, cfg.mlp_ratio, operations, **kw)
        self.mlp_post_norm = operations.RMSNorm(dim, eps=eps, **kw)
        self.adaln = SharedBias(dim, **kw)

    def forward(self, x, y, img_mod, txt_mod, rope_img, rope_txt, transformer_options):
        batch, n_text = x.shape[0], y.shape[1]
        tokens = torch.cat((y, x), 1)
        s1, c1, g1, s2, c2, g2 = self.adaln(img_mod).chunk(6, -1)
        h = modulate(self.norm1(tokens), s1, c1)
        q = self.q_proj(h).reshape(batch, -1, self.num_heads, self.head_dim)
        k = self.k_proj(h).reshape(batch, -1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(h).reshape(batch, -1, self.num_kv_heads, self.head_dim)
        q, k = norm_rope(q, k, self.q_norm, self.k_norm, torch.cat((rope_txt, rope_img), dim=2))
        attn = attention(q, k, v, transformer_options=transformer_options)
        attn = self.attn_post_norm(self.attn_proj(attn * self.attn_gate(h).sigmoid()))
        tokens = torch.addcmul(tokens, g1, attn)
        tokens = torch.addcmul(tokens, g2, self.mlp_post_norm(self.mlp(modulate(self.norm2(tokens), s2, c2))))
        return tokens[:, n_text:], tokens[:, :n_text]


class PixelBlock(nn.Module):
    def __init__(self, cfg, operations, **kw):
        super().__init__()
        dim, count = cfg.pixel_hidden_size, cfg.patch_size ** 2
        self.count = count
        self.norm1 = operations.RMSNorm(dim, eps=cfg.norm_eps, **kw)
        self.norm2 = operations.RMSNorm(dim, eps=cfg.norm_eps, **kw)
        self.adaln = operations.Linear(cfg.hidden_size, 4 * dim * count, **kw)
        self.compress = operations.Linear(count * dim, cfg.pixel_attn_hidden_size, **kw)
        self.expand = operations.Linear(cfg.pixel_attn_hidden_size, count * dim, **kw)
        self.attn = SelfAttention(cfg.pixel_attn_hidden_size, cfg.pixel_num_heads, cfg.norm_eps, operations, **kw)
        self.mlp = nn.Module()
        self.mlp.fc1 = operations.Linear(dim, int(dim * cfg.pixel_mlp_ratio), **kw)
        self.mlp.fc2 = operations.Linear(int(dim * cfg.pixel_mlp_ratio), dim, **kw)

    def forward(self, x, cond, rope, grid, transformer_options):
        count = grid[0] * grid[1]
        scale1, shift1, scale2, shift2 = self.adaln(cond).reshape(x.shape[0], self.count, -1).chunk(4, -1)
        compact = self.compress(self.norm1(x).flatten(1)).reshape(-1, count, self.compress.out_features)
        branch = self.attn(compact, rope, transformer_options=transformer_options)
        branch = self.expand(branch.flatten(0, 1)).reshape_as(x)
        x = x + modulate(branch, shift1, scale1)
        branch = self.mlp.fc2(F.gelu(self.mlp.fc1(self.norm2(x))))
        return x + modulate(branch, shift2, scale2)


class IrisDiT(nn.Module):
    def __init__(self, cfg=None, device=None, dtype=None, operations=comfy.ops.manual_cast):
        super().__init__()
        self.cfg = cfg or IrisConfig()
        self.dtype = dtype
        cfg = self.cfg
        kw = {"device": device, "dtype": dtype}
        self.s_embedder = nn.Module()
        self.s_embedder.proj = operations.Linear(cfg.patch_size ** 2 * cfg.in_channels, cfg.hidden_size, **kw)
        self.t_embedder = nn.Module()
        self.t_embedder.mlp = nn.Sequential(operations.Linear(256, cfg.hidden_size, **kw), nn.SiLU(), operations.Linear(cfg.hidden_size, cfg.hidden_size, **kw))
        self.y_embedder = LayerwiseTextEmbedder(cfg, operations, **kw)
        self.y_pos_embedding = nn.Parameter(torch.empty(1, cfg.text_len, cfg.hidden_size, **kw))
        self.modulation_cores = nn.ModuleDict({f"adaln_{stream}": operations.Linear(cfg.hidden_size, 6 * cfg.hidden_size, **kw) for stream in ("img", "txt")})
        self.blocks = nn.ModuleList([(DualBlock if i < cfg.dual_depth else SingleBlock)(cfg, operations, **kw) for i in range(cfg.depth)])
        self.pixel_embedder = nn.Module()
        self.pixel_embedder.proj = operations.Linear(cfg.in_channels, cfg.pixel_hidden_size, **kw)
        self.pixel_blocks = nn.ModuleList([PixelBlock(cfg, operations, **kw) for _ in range(cfg.pixel_depth)])
        self.final_layer = nn.Module()
        self.final_layer.norm = operations.RMSNorm(cfg.pixel_hidden_size, eps=cfg.norm_eps, **kw)
        self.final_layer.linear = operations.Linear(cfg.pixel_hidden_size, cfg.in_channels, **kw)

    def preprocess_text(self, context, mask):
        y = self.y_embedder(context[:, :self.cfg.text_len], mask[:, :self.cfg.text_len])
        return y + comfy.ops.cast_to_input(self.y_pos_embedding, y)[:, :y.shape[1]]

    def forward(self, x, timestep, context, control=None, transformer_options={}, **kwargs):
        cfg, p = self.cfg, self.cfg.patch_size
        height, width = x.shape[-2:]
        x = pad_to_patch_size(x, (p, p), padding_mode="replicate")
        batch, _, padded_h, padded_w = x.shape
        grid = padded_h // p, padded_w // p
        count = grid[0] * grid[1]
        s = self.s_embedder.proj(F.unfold(x, kernel_size=p, stride=p).transpose(1, 2))
        freq = torch.exp(-math.log(cfg.timestep_max_period) * torch.arange(128, device=x.device, dtype=torch.float32) / 128)
        phase = timestep.float().reshape(-1, 1) * freq
        t = self.t_embedder.mlp(torch.cat((phase.cos(), phase.sin()), -1).to(x.dtype)).unsqueeze(1)
        cond = F.silu(t)
        img_mod, txt_mod = self.modulation_cores.adaln_img(cond), self.modulation_cores.adaln_txt(cond)
        rope_img = image_rope(cfg.hidden_size // cfg.num_heads, grid, cfg, x.device)
        rope_txt = text_rope(cfg.hidden_size // cfg.num_heads, context.shape[1], cfg.text_rope_theta, x.device)
        y = context
        prefetch = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), x.device, transformer_options)
        for block in self.blocks:
            def core():
                nonlocal s, y
                s, y = block(s, y, img_mod, txt_mod, rope_img, rope_txt, transformer_options)
            comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, block, x.dtype, core=core, malloc_scope="block")
        comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, None, malloc_scope="block")
        s = F.silu(t + s).reshape(batch * count, -1)
        pixels = self.pixel_embedder.proj(x.permute(0, 2, 3, 1))
        pixels = pixels + pixel_position(cfg.pixel_hidden_size, padded_h, padded_w, x.device, pixels.dtype)
        pixels = pixels.reshape(batch, grid[0], p, grid[1], p, -1).permute(0, 1, 3, 2, 4, 5).reshape(batch * count, p * p, -1)
        rope_pix = image_rope(cfg.pixel_attn_hidden_size // cfg.pixel_num_heads, grid, cfg, x.device)
        for block in self.pixel_blocks:
            pixels = block(pixels, s, rope_pix, grid, transformer_options)
        out = self.final_layer.linear(self.final_layer.norm(pixels))
        folded = out.reshape(batch, count, p * p, cfg.in_channels).permute(0, 3, 2, 1).reshape(batch, -1, count)
        return F.fold(folded, (padded_h, padded_w), kernel_size=p, stride=p)[:, :, :height, :width]
