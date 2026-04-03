"""
DiT backbone with per-block style cross-attention, built on top of F5-TTS.

The idea is simple: take the pretrained DiT, freeze it, then inject a
cross-attention layer before each transformer block that attends to a
learned style embedding. Output projections are zero-initialized so the
model starts from the exact pretrained behavior and gradually learns to
incorporate style information.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    AdaLayerNorm_Final, ConvNeXtV2Block, DiTBlock,
    TimestepEmbedding, precompute_freqs_cis,
)
from f5_tts.model.backbones.dit import TextEmbedding, InputEmbedding


class StyleEmbedding(nn.Module):

    def __init__(self, num_styles, style_dim, conv_layers=4, conv_mult=2):
        super().__init__()
        self.style_embed = nn.Embedding(num_styles + 1, style_dim)  # 0 = null token
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(style_dim, 8192), persistent=False,
        )
        self.convs = nn.Sequential(
            *[ConvNeXtV2Block(style_dim, style_dim * conv_mult) for _ in range(conv_layers)]
        )

    def forward(self, style, seq_len, drop_style=False):
        style = style[:, :seq_len]
        if style.shape[1] < seq_len:
            style = F.pad(style, (0, seq_len - style.shape[1]), value=0)

        if drop_style:
            style = torch.zeros_like(style)

        x = self.style_embed(style) + self.freqs_cis[:seq_len]
        for conv in self.convs:
            x = conv(x)
        return x


class StyleCrossAttention(nn.Module):
    """Queries from the main stream, K/V from style embeddings."""

    def __init__(self, dim, style_dim, heads=8, dim_head=64):
        super().__init__()
        inner = heads * dim_head
        self.heads, self.dim_head = heads, dim_head

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(style_dim, inner, bias=False)
        self.to_v = nn.Linear(style_dim, inner, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)
        nn.init.zeros_(self.to_out.weight)  # starts as identity

    def forward(self, x, style_embed):
        B, N, _ = x.shape
        h, d = self.heads, self.dim_head

        q = self.to_q(self.norm(x)).view(B, N, h, d).transpose(1, 2)
        k = self.to_k(style_embed).view(B, N, h, d).transpose(1, 2)
        v = self.to_v(style_embed).view(B, N, h, d).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return x + self.to_out(out)


class DiTStyled(nn.Module):
    """
    F5-TTS DiT + style cross-attention at every block.

    All the base DiT parameters (time_embed, text_embed, input_embed,
    transformer_blocks, norm_out, proj_out) are loaded from a pretrained
    checkpoint. The style_embed and style_cross_attn_blocks are new and
    trained from scratch.
    """

    def __init__(
        self, *, dim, depth=8, heads=8, dim_head=64, dropout=0.1, ff_mult=4,
        mel_dim=100, text_num_embeds=256, text_dim=None, text_mask_padding=True,
        qk_norm=None, conv_layers=0, pe_attn_head=None, attn_backend="torch",
        attn_mask_enabled=False, long_skip_connection=False,
        checkpoint_activations=False,
        num_styles=9, style_dim=None, style_conv_layers=4,
        style_cross_attn_heads=8, style_cross_attn_dim_head=64,
    ):
        super().__init__()

        if text_dim is None:
            text_dim = mel_dim
        if style_dim is None:
            style_dim = text_dim

        self.dim = dim
        self.depth = depth
        self.checkpoint_activations = checkpoint_activations

        # base DiT components (pretrained)
        self.time_embed = TimestepEmbedding(dim)
        self.text_embed = TextEmbedding(
            text_num_embeds, text_dim, mask_padding=text_mask_padding,
            conv_layers=conv_layers,
        )
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.transformer_blocks = nn.ModuleList([
            DiTBlock(
                dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult,
                dropout=dropout, qk_norm=qk_norm, pe_attn_head=pe_attn_head,
                attn_backend=attn_backend, attn_mask_enabled=attn_mask_enabled,
            )
            for _ in range(depth)
        ])
        self.long_skip_connection = (
            nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        )
        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

        # style conditioning (new, trained from scratch)
        self.style_embed = StyleEmbedding(num_styles, style_dim, style_conv_layers)
        self.style_cross_attn_blocks = nn.ModuleList([
            StyleCrossAttention(dim, style_dim, style_cross_attn_heads, style_cross_attn_dim_head)
            for _ in range(depth)
        ])

        # text embed cache for inference
        self._text_cond = None
        self._text_uncond = None

        self._zero_init_base()

    def _zero_init_base(self):
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def _get_input_embed(self, x, cond, text, drop_audio_cond=False,
                         drop_text=False, cache=True, audio_mask=None):
        if self._text_uncond is None or self._text_cond is None or not cache:
            seq_len = x.shape[1] if audio_mask is None else audio_mask.sum(dim=1)
            text_embed = self.text_embed(text, seq_len=seq_len, drop_text=drop_text)
            if cache:
                if drop_text:
                    self._text_uncond = text_embed
                else:
                    self._text_cond = text_embed

        if cache:
            text_embed = self._text_uncond if drop_text else self._text_cond

        return self.input_embed(x, cond, text_embed,
                                drop_audio_cond=drop_audio_cond, audio_mask=audio_mask)

    def clear_cache(self):
        self._text_cond = self._text_uncond = None

    def _run_blocks(self, x, t, style_emb, mask, rope):
        if self.long_skip_connection is not None:
            residual = x

        for i, block in enumerate(self.transformer_blocks):
            x = self.style_cross_attn_blocks[i](x, style_emb)
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, t, mask, rope, use_reentrant=False,
                )
            else:
                x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        return self.proj_out(self.norm_out(x, t))

    def forward(self, x, cond, text, time, style, mask=None,
                drop_audio_cond=False, drop_text=False, drop_style=False,
                cfg_infer=False, cache=False):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)
        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if cfg_infer:
            # pack 3 variants for multi-term CFG: full / uncond / no-style
            x_full = self._get_input_embed(x, cond, text, cache=cache, audio_mask=mask)
            x_uncond = self._get_input_embed(x, cond, text, drop_audio_cond=True,
                                             drop_text=True, cache=cache, audio_mask=mask)
            x_nostyle = self._get_input_embed(x, cond, text, cache=cache, audio_mask=mask)

            x = torch.cat((x_full, x_uncond, x_nostyle), dim=0)
            t = t.repeat(3, 1)
            mask = mask.repeat(3, 1) if mask is not None else None

            s_cond = self.style_embed(style, seq_len, drop_style=False)
            s_null = self.style_embed(style, seq_len, drop_style=True)
            style_emb = torch.cat((s_cond, s_null, s_null), dim=0)

            return self._run_blocks(x, t, style_emb, mask, rope)

        x = self._get_input_embed(x, cond, text, drop_audio_cond=drop_audio_cond,
                                  drop_text=drop_text, cache=cache, audio_mask=mask)
        style_emb = self.style_embed(style, seq_len, drop_style=drop_style)
        return self._run_blocks(x, t, style_emb, mask, rope)
