"""Dataset and batching utilities for SaturnTTS training."""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler


class SaturnDataset(Dataset):

    def __init__(self, metadata_path, styles_path, data_root,
                 min_duration=0.5, max_duration=30.0):
        self.data_root = Path(data_root)

        with open(metadata_path) as f:
            self.metadata = json.load(f)
        with open(styles_path) as f:
            self.styles_map = json.load(f)

        self.metadata = [
            m for m in self.metadata
            if min_duration <= m["duration"] <= max_duration
        ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        entry = self.metadata[idx]

        mel = torch.load(self.data_root / entry["mel_path"], weights_only=True)
        mel = mel.permute(1, 0)  # [T, n_mels]

        # style IDs are offset by 1 so 0 can serve as the null token
        style_id = self.styles_map[entry["style"]] + 1
        mel_frames = mel.shape[0]

        return {
            "mel": mel,
            "text": entry["text"],
            "style": torch.full((mel_frames,), style_id, dtype=torch.long),
            "mel_frames": mel_frames,
            "id": entry["id"],
        }


class DynamicBatchSampler(Sampler):
    """Packs samples by duration to minimize padding waste."""

    def __init__(self, dataset, max_frames=12000, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.max_frames = max_frames
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.durations = [m["mel_frames"] for m in dataset.metadata]

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            indices.sort(key=lambda i: self.durations[i])
            buckets = [indices[i:i+100] for i in range(0, len(indices), 100)]
            random.shuffle(buckets)
            indices = [idx for b in buckets for idx in b]

        batches = []
        batch = []
        max_in_batch = 0

        for idx in indices:
            frames = self.durations[idx]
            new_max = max(max_in_batch, frames)
            if batch and new_max * (len(batch) + 1) > self.max_frames:
                batches.append(batch)
                batch, max_in_batch = [idx], frames
            else:
                batch.append(idx)
                max_in_batch = new_max

        if batch and not self.drop_last:
            batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self):
        return max(1, sum(self.durations) // self.max_frames)


def collate_fn(batch):
    mels = [b["mel"] for b in batch]
    styles = [b["style"] for b in batch]
    mel_frames = torch.tensor([b["mel_frames"] for b in batch], dtype=torch.long)
    max_len = mel_frames.max().item()
    mel_dim = mels[0].shape[1]

    mel_padded = torch.zeros(len(batch), max_len, mel_dim)
    style_padded = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i in range(len(batch)):
        n = mels[i].shape[0]
        mel_padded[i, :n] = mels[i]
        style_padded[i, :n] = styles[i]

    return {
        "mel": mel_padded,
        "text": [b["text"] for b in batch],
        "style": style_padded,
        "mel_lens": mel_frames,
    }
