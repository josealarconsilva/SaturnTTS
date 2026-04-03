#!/usr/bin/env python3
"""
Phase A: vanilla fine-tune of the pretrained F5-TTS on Expresso.
No style conditioning here -- just domain adaptation to the new speakers
and recording environment. Typically 5-20K steps is enough.
"""

import sys
sys.path.insert(0, "/home/jose/SaturnTTS/src")

import math
from pathlib import Path

import torch
import yaml
from ema_pytorch import EMA
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from f5_tts.model.cfm import CFM
from f5_tts.model.backbones.dit import DiT
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import list_str_to_idx

from saturn_tts.model.dataset import SaturnDataset, DynamicBatchSampler, collate_fn

ROOT = Path("/home/jose/SaturnTTS")


def load_vocab(path):
    with open(path) as f:
        return {line.rstrip("\n"): i for i, line in enumerate(f)}


def cosine_lr(step, warmup, total, peak):
    if step < warmup:
        return peak * step / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * max(0.0, 0.5 * (1 + math.cos(math.pi * t)))


def main():
    with open(ROOT / "configs/model.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open(ROOT / "configs/train_a.yaml") as f:
        train_cfg = yaml.safe_load(f)

    device = torch.device("cuda")
    mc, tc, dc = model_cfg["model"], train_cfg["training"], train_cfg["data"]

    dit = DiT(
        dim=mc["dim"], depth=mc["depth"], heads=mc["heads"],
        dim_head=mc["dim_head"], dropout=mc["dropout"], ff_mult=mc["ff_mult"],
        mel_dim=mc["mel_dim"], text_num_embeds=mc["text_num_embeds"],
        text_dim=mc["text_dim"], text_mask_padding=mc["text_mask_padding"],
        conv_layers=mc["conv_layers"],
        checkpoint_activations=mc["checkpoint_activations"],
    )

    # load pretrained F5-TTS (keys are under "ema_model.transformer.")
    state = load_file(train_cfg["pretrained"]["checkpoint"])
    prefix = "ema_model.transformer."
    mapped = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    dit.load_state_dict(mapped, strict=True)
    print(f"loaded {len(mapped)} pretrained keys")

    vocab = load_vocab(ROOT / "data/vocab_f5.txt")

    mel_spec = MelSpec(
        n_fft=model_cfg["mel"]["n_fft"], hop_length=model_cfg["mel"]["hop_length"],
        win_length=model_cfg["mel"]["win_length"],
        n_mel_channels=model_cfg["mel"]["n_mel_channels"],
        target_sample_rate=model_cfg["mel"]["target_sample_rate"],
        mel_spec_type=model_cfg["mel"]["mel_spec_type"],
    )
    cfm = CFM(
        transformer=dit, sigma=model_cfg["cfm"]["sigma"],
        audio_drop_prob=model_cfg["cfm"]["audio_drop_prob"], cond_drop_prob=0.2,
        mel_spec_module=mel_spec,
        frac_lengths_mask=tuple(model_cfg["cfm"]["frac_lengths_mask"]),
        vocab_char_map=vocab,
    ).to(device)

    ema = EMA(cfm, beta=tc["ema_decay"])

    ds = SaturnDataset(dc["train_metadata"], dc["styles_path"], dc["data_root"])
    sampler = DynamicBatchSampler(ds, max_frames=tc["max_frames_per_batch"])
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn,
                        num_workers=4, pin_memory=True)

    opt = torch.optim.AdamW(cfm.parameters(), lr=tc["learning_rate"],
                            weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    save_dir = ROOT / "checkpoints/phase_a"
    save_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    max_steps = tc["max_steps"]
    accum = tc["grad_accumulation_steps"]
    running_loss = 0.0

    print(f"phase A: {max_steps} steps, {len(ds)} samples")

    cfm.train()
    while step < max_steps:
        for batch in loader:
            mel = batch["mel"].to(device)
            text = list_str_to_idx(batch["text"], vocab).to(device)
            lens = batch["mel_lens"].to(device)

            lr = cosine_lr(step, tc["warmup_steps"], max_steps, tc["learning_rate"])
            for pg in opt.param_groups:
                pg["lr"] = lr

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, _, _ = cfm(inp=mel, text=text, lens=lens)

            scaler.scale(loss / accum).backward()
            running_loss += loss.item()

            if (step + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(cfm.parameters(), tc["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                ema.update()

            step += 1

            if step % tc["log_every"] == 0:
                print(f"step {step}/{max_steps}  loss={running_loss/tc['log_every']:.4f}  lr={lr:.2e}")
                running_loss = 0.0

            if step % tc["save_every"] == 0:
                ckpt = {"step": step, "model_state_dict": cfm.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "ema_state_dict": ema.state_dict()}
                torch.save(ckpt, save_dir / f"step_{step}.pt")
                torch.save(ckpt, save_dir / "latest.pt")
                # keep only recent checkpoints
                for old in sorted(save_dir.glob("step_*.pt"),
                                  key=lambda p: int(p.stem.split("_")[1]))[:-tc["keep_last_n_checkpoints"]]:
                    old.unlink()

            if step >= max_steps:
                break

    torch.save({"step": step, "model_state_dict": cfm.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "ema_state_dict": ema.state_dict()}, save_dir / "latest.pt")
    print(f"phase A done at step {step}")


if __name__ == "__main__":
    main()
