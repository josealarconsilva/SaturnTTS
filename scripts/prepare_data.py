#!/usr/bin/env python3
"""
Extracts audio from the Expresso parquet shards, resamples to 24kHz,
precomputes mel spectrograms, and writes train/val metadata splits.
"""

import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio

ROOT = Path("/home/jose/SaturnTTS")
PARQUET_DIR = ROOT / "expresso" / "read"
WAV_DIR = ROOT / "data" / "wavs"
MEL_DIR = ROOT / "data" / "mels"
DATA_DIR = ROOT / "data"

TARGET_SR = 24_000
SKIP_STYLES = {"singing", "longform"}  # too few samples


def compute_mel(wav):
    """Vocos-compatible log-mel spectrogram (matches F5-TTS)."""
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR, n_fft=1024, win_length=1024,
        hop_length=256, n_mels=100, power=1, center=True,
        normalized=False, norm=None,
    )
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return transform(wav).clamp(min=1e-5).log().squeeze(0)


def main():
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    MEL_DIR.mkdir(parents=True, exist_ok=True)

    shards = sorted(PARQUET_DIR.glob("train-*.parquet"))
    if not shards:
        print("no parquet files found"); sys.exit(1)

    metadata = []
    style_set, vocab_chars = set(), set()
    skipped = processed = 0

    for pf in shards:
        print(f"  {pf.name}")
        table = pq.read_table(pf)
        rows = table.to_pydict()

        for i in range(len(table)):
            sid, style = rows["id"][i], rows["style"][i]
            if style in SKIP_STYLES:
                skipped += 1; continue

            wav_path = WAV_DIR / f"{sid}.wav"
            mel_path = MEL_DIR / f"{sid}.pt"

            if not wav_path.exists():
                audio_bytes = rows["audio"][i]["bytes"]
                wav_data, orig_sr = sf.read(io.BytesIO(audio_bytes))
                wav_tensor = torch.from_numpy(wav_data).float().unsqueeze(0)
                if orig_sr != TARGET_SR:
                    wav_tensor = torchaudio.functional.resample(wav_tensor, orig_sr, TARGET_SR)
                sf.write(str(wav_path), wav_tensor.squeeze(0).numpy(), TARGET_SR)
            else:
                wav_tensor = None

            if not mel_path.exists():
                if wav_tensor is None:
                    wav_data, _ = sf.read(str(wav_path))
                    wav_tensor = torch.from_numpy(wav_data).float().unsqueeze(0)
                mel = compute_mel(wav_tensor.squeeze(0))
                torch.save(mel, mel_path)
            else:
                mel = torch.load(mel_path, weights_only=True)

            duration = mel.shape[1] * 256 / TARGET_SR
            style_set.add(style)
            vocab_chars.update(rows["text"][i])

            metadata.append({
                "id": sid,
                "audio_path": f"wavs/{sid}.wav",
                "mel_path": f"mels/{sid}.pt",
                "text": rows["text"][i],
                "speaker_id": rows["speaker_id"][i],
                "style": style,
                "duration": round(duration, 3),
                "mel_frames": int(mel.shape[1]),
            })
            processed += 1
            if processed % 1000 == 0:
                print(f"    {processed} done")

    print(f"\n{processed} processed, {skipped} skipped")

    styles = sorted(style_set)
    with open(DATA_DIR / "styles.json", "w") as f:
        json.dump({s: i for i, s in enumerate(styles)}, f, indent=2)
    with open(DATA_DIR / "vocab.txt", "w") as f:
        f.writelines(ch + "\n" for ch in sorted(vocab_chars))

    # 95/5 stratified split
    groups = defaultdict(list)
    for e in metadata:
        groups[(e["speaker_id"], e["style"])].append(e)

    train, val = [], []
    for items in groups.values():
        n_val = max(1, int(len(items) * 0.05))
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    for name, data in [("metadata", metadata), ("metadata_train", train), ("metadata_val", val)]:
        with open(DATA_DIR / f"{name}.json", "w") as f:
            json.dump(data, f)

    total_h = sum(e["duration"] for e in metadata) / 3600
    print(f"styles: {styles}")
    print(f"train: {len(train)}, val: {len(val)}, total: {total_h:.2f}h")


if __name__ == "__main__":
    main()
