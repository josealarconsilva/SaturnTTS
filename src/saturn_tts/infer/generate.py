"""
SaturnTTS inference: text + reference audio + style tag -> speech.

The key trick is multi-term CFG with separate guidance scales for
content (cfg_strength) and emotion (style_cfg_strength), letting you
dial emotional intensity independently from overall quality.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from vocos import Vocos

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import list_str_to_idx

from saturn_tts.model.backbones.dit_style import DiTStyled
from saturn_tts.model.cfm_style import CFMStyled


def load_model(checkpoint_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    mc = cfg["model"]

    dit = DiTStyled(
        dim=mc["dim"], depth=mc["depth"], heads=mc["heads"],
        dim_head=mc["dim_head"], dropout=0.0, ff_mult=mc["ff_mult"],
        mel_dim=mc["mel_dim"], text_num_embeds=mc["text_num_embeds"],
        text_dim=mc["text_dim"], text_mask_padding=mc["text_mask_padding"],
        conv_layers=mc["conv_layers"], checkpoint_activations=False,
        num_styles=mc["num_styles"], style_dim=mc["style_dim"],
        style_conv_layers=mc["style_conv_layers"],
        style_cross_attn_heads=mc["style_cross_attn_heads"],
        style_cross_attn_dim_head=mc["style_cross_attn_dim_head"],
    )

    mel_spec = MelSpec(
        n_fft=cfg["mel"]["n_fft"], hop_length=cfg["mel"]["hop_length"],
        win_length=cfg["mel"]["win_length"],
        n_mel_channels=cfg["mel"]["n_mel_channels"],
        target_sample_rate=cfg["mel"]["target_sample_rate"],
        mel_spec_type=cfg["mel"]["mel_spec_type"],
    )

    cfm = CFMStyled(
        transformer=dit, sigma=cfg["cfm"]["sigma"],
        mel_spec_module=mel_spec,
        frac_lengths_mask=tuple(cfg["cfm"]["frac_lengths_mask"]),
    )

    # prefer EMA weights when available
    if "ema_state_dict" in ckpt:
        ema_state = {}
        for k, v in ckpt["ema_state_dict"].items():
            if k.startswith("ema_model."):
                ema_state[k[len("ema_model."):]] = v
        if ema_state:
            cfm.load_state_dict(ema_state)
        else:
            cfm.load_state_dict(ckpt["model_state_dict"])
    else:
        cfm.load_state_dict(ckpt["model_state_dict"])

    return cfm.to(device).eval(), cfg


def load_vocoder(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()


def generate(model, vocoder, text, ref_audio_path, ref_text, style,
             styles_map, vocab_char_map=None, *,
             cfg_strength=2.0, style_cfg_strength=5.0, steps=32,
             sway_sampling_coef=-1.0, speed=1.0, seed=None, device=None):
    if device is None:
        device = next(model.parameters()).device

    # load reference audio
    ref_data, ref_sr = sf.read(str(ref_audio_path))
    ref_wav = torch.from_numpy(ref_data).float()
    if ref_wav.dim() == 1:
        ref_wav = ref_wav.unsqueeze(0)
    if ref_sr != 24000:
        ref_wav = torchaudio.functional.resample(ref_wav, ref_sr, 24000)
    ref_wav = ref_wav.mean(0, keepdim=True).to(device)

    ref_mel = model.mel_spec(ref_wav).permute(0, 2, 1)
    ref_len = ref_mel.shape[1]

    # estimate output duration from the ratio of ref audio to ref text
    ref_text_len = max(len(ref_text.encode("utf-8")), 1)
    gen_text_len = len(text.encode("utf-8"))
    gen_frames = max(int(ref_len / ref_text_len * gen_text_len / speed), 10)
    total = ref_len + gen_frames

    full_text = ref_text + " " + text
    text_ids = list_str_to_idx([full_text], vocab_char_map).to(device)

    style_id = styles_map[style] + 1
    style_tensor = torch.full((1, total), style_id, dtype=torch.long, device=device)
    lens = torch.tensor([ref_len], device=device, dtype=torch.long)

    with torch.no_grad():
        out, _ = model.sample(
            cond=ref_mel, text=text_ids,
            duration=torch.tensor([total], device=device, dtype=torch.long),
            style=style_tensor, lens=lens,
            steps=steps, cfg_strength=cfg_strength,
            style_cfg_strength=style_cfg_strength,
            sway_sampling_coef=sway_sampling_coef, seed=seed,
            vocoder=lambda mel: vocoder.decode(mel),
        )

    # trim the reference portion
    ref_samples = ref_len * 256
    if out.shape[-1] > ref_samples:
        out = out[:, ref_samples:]

    return out
