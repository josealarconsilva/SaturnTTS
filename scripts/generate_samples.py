#!/usr/bin/env python3
"""Batch-generate samples across styles for quick evaluation."""

import sys
sys.path.insert(0, "/home/jose/SaturnTTS/src")

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from saturn_tts.infer.generate import load_model, load_vocoder, generate

ROOT = Path("/home/jose/SaturnTTS")
OUT = ROOT / "outputs/samples"
OUT.mkdir(parents=True, exist_ok=True)

CHECKPOINT = ROOT / "checkpoints/phase_b/latest.pt"
REF_AUDIO = ROOT / "data/shadowheart_ref.wav"
REF_TEXT = "You Selunites are coddled. Decadent. Drunk on hope."

SENTENCES = [
    "This is absolutely incredible news, I can not believe it!",
    "I am not sure I understand what you are trying to say.",
    "The weather today is quite pleasant for a walk in the park.",
]
STYLES = ["default", "happy", "sad", "confused", "whisper", "laughing"]
INTENSITIES = [0.0, 3.0, 7.0]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(ROOT / "data/styles.json") as f:
        styles = json.load(f)
    with open(ROOT / "data/vocab_f5.txt") as f:
        vocab = {line.rstrip("\n"): i for i, line in enumerate(f)}

    model, _ = load_model(CHECKPOINT, device)
    vocoder = load_vocoder(device)

    n = 0
    for text in SENTENCES:
        slug = text[:40].replace(" ", "_").translate(str.maketrans("", "", ",'!"))
        for style in STYLES:
            if style not in styles:
                continue
            for scfg in INTENSITIES:
                fname = f"{style}_s{scfg:.0f}_{slug}.wav"
                wav = generate(
                    model, vocoder, text=text, ref_audio_path=REF_AUDIO,
                    ref_text=REF_TEXT, style=style, styles_map=styles,
                    vocab_char_map=vocab, cfg_strength=2.0,
                    style_cfg_strength=scfg, steps=32, seed=42, device=device,
                )
                audio = wav.cpu().squeeze().numpy()
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio *= 0.7 / peak
                sf.write(str(OUT / fname), audio, 24000)
                print(f"  {fname} ({len(audio)/24000:.1f}s)")
                n += 1

    print(f"\n{n} samples in {OUT}")


if __name__ == "__main__":
    main()
