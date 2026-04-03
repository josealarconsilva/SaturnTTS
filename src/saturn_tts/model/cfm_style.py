"""
Flow matching with multi-term classifier-free guidance for style control.

Training uses three-way dropout:
  - 15% fully unconditional (drop everything)
  - 15% style-unconditional (keep audio+text, drop style)
  - 70% fully conditional (with independent audio dropout at 30%)

Inference uses dual guidance scales to separately control overall
conditioning strength and emotional intensity.
"""

from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default, exists, lens_to_mask,
    list_str_to_idx, list_str_to_tensor, mask_from_frac_lengths,
)


class CFMStyled(nn.Module):

    def __init__(self, transformer, sigma=0.0,
                 odeint_kwargs=dict(method="euler"),
                 audio_drop_prob=0.3, cond_drop_prob=0.15,
                 style_drop_prob=0.15, num_channels=None,
                 mel_spec_module=None, mel_spec_kwargs=dict(),
                 frac_lengths_mask=(0.7, 1.0), vocab_char_map=None):
        super().__init__()

        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        self.num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.transformer = transformer
        self.dim = transformer.dim

        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.style_drop_prob = style_drop_prob
        self.frac_lengths_mask = frac_lengths_mask
        self.sigma = sigma
        self.odeint_kwargs = odeint_kwargs
        self.vocab_char_map = vocab_char_map

    @property
    def device(self):
        return next(self.parameters()).device

    def _to_mel(self, x):
        if x.ndim == 2:
            x = self.mel_spec(x).permute(0, 2, 1)
        return x

    def _tokenize(self, text, device):
        if isinstance(text, list):
            if self.vocab_char_map:
                return list_str_to_idx(text, self.vocab_char_map).to(device)
            return list_str_to_tensor(text).to(device)
        return text

    @torch.no_grad()
    def sample(self, cond, text, duration, style, *, lens=None, steps=32,
               cfg_strength=2.0, style_cfg_strength=5.0,
               sway_sampling_coef=None, seed=None, max_duration=65536,
               vocoder=None, no_ref_audio=False):
        self.eval()

        cond = self._to_mel(cond).to(next(self.parameters()).dtype)
        batch, cond_seq_len, device = *cond.shape[:2], cond.device

        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        text = self._tokenize(text, device)

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)
        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration,
        ).clamp(max=max_duration)
        max_dur = duration.amax()

        cond = F.pad(cond, (0, 0, 0, max_dur - cond_seq_len))
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(lens_to_mask(lens), (0, max_dur - lens.shape[-1] if lens.ndim == 1 else 0), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

        if style.shape[1] < max_dur:
            style = F.pad(style, (0, max_dur - style.shape[1]),
                          value=style[:, 0:1].squeeze(-1))

        mask = lens_to_mask(duration) if batch > 1 else None
        use_cfg = cfg_strength > 1e-5 or style_cfg_strength > 1e-5

        def ode_fn(t, x):
            if not use_cfg:
                return self.transformer(
                    x=x, cond=step_cond, text=text, time=t, style=style,
                    mask=mask,
                )

            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, style=style,
                mask=mask, cfg_infer=True,
            )
            v_full, v_null, v_nostyle = torch.chunk(pred, 3, dim=0)

            return (v_full
                    + cfg_strength * (v_full - v_null)
                    + style_cfg_strength * (v_full - v_nostyle))

        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=device,
                                  dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t = torch.linspace(0, 1, steps + 1, device=device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(ode_fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        out = torch.where(cond_mask, cond, trajectory[-1])

        if exists(vocoder):
            out = vocoder(out.permute(0, 2, 1))

        return out, trajectory

    def forward(self, inp, text, style, *, lens=None, noise_scheduler=None):
        inp = self._to_mel(inp)
        batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device

        text = self._tokenize(text, device)

        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)

        frac_lengths = torch.zeros((batch,), device=device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)
        if exists(mask):
            rand_span_mask &= mask

        # flow matching: linear interpolation between noise and target
        x1 = inp
        x0 = torch.randn_like(x1)
        t = torch.rand((batch,), dtype=dtype, device=device)
        phi = (1 - t.unsqueeze(-1).unsqueeze(-1)) * x0 + t.unsqueeze(-1).unsqueeze(-1) * x1
        flow = x1 - x0

        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # three-way CFG dropout
        r = random()
        if r < self.cond_drop_prob:
            drop_audio, drop_text, drop_style = True, True, True
        elif r < self.cond_drop_prob + self.style_drop_prob:
            drop_audio, drop_text, drop_style = False, False, True
        else:
            drop_audio = random() < self.audio_drop_prob
            drop_text, drop_style = False, False

        if style.shape[1] < seq_len:
            style = F.pad(style, (0, seq_len - style.shape[1]),
                          value=style[:, 0:1].squeeze(-1))
        style = style[:, :seq_len]

        pred = self.transformer(
            x=phi, cond=cond, text=text, time=t, style=style,
            mask=mask, drop_audio_cond=drop_audio,
            drop_text=drop_text, drop_style=drop_style,
        )

        loss = F.mse_loss(pred, flow, reduction="none")[rand_span_mask].mean()
        return loss, cond, pred
