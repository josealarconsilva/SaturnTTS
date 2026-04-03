#!/usr/bin/env python3
"""
Phase B: style-conditioned training with multi-term CFG.

Loads the Phase A checkpoint into DiTStyled (which adds style cross-attention
blocks), then trains in two stages:
  1. First 10K steps: freeze base DiT, train only style modules
  2. After 10K: unfreeze everything, end-to-end with lower LR on base params
"""

import sys
sys.path.insert(0, "/home/jose/SaturnTTS/src")

import math
from pathlib import Path

import torch
import yaml
from ema_pytorch import EMA
from torch.utils.data import DataLoader

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import list_str_to_idx

from saturn_tts.model.backbones.dit_style import DiTStyled
from saturn_tts.model.cfm_style import CFMStyled
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
    with open(ROOT / "configs/train_b.yaml") as f:
        train_cfg = yaml.safe_load(f)

    device = torch.device("cuda")
    mc, tc, dc = model_cfg["model"], train_cfg["training"], train_cfg["data"]

    dit = DiTStyled(
        dim=mc["dim"], depth=mc["depth"], heads=mc["heads"],
        dim_head=mc["dim_head"], dropout=mc["dropout"], ff_mult=mc["ff_mult"],
        mel_dim=mc["mel_dim"], text_num_embeds=mc["text_num_embeds"],
        text_dim=mc["text_dim"], text_mask_padding=mc["text_mask_padding"],
        conv_layers=mc["conv_layers"],
        checkpoint_activations=mc["checkpoint_activations"],
        num_styles=mc["num_styles"], style_dim=mc["style_dim"],
        style_conv_layers=mc["style_conv_layers"],
        style_cross_attn_heads=mc["style_cross_attn_heads"],
        style_cross_attn_dim_head=mc["style_cross_attn_dim_head"],
    )

    # load phase A weights into the base DiT components
    phase_a = torch.load(tc.get("phase_a_checkpoint", train_cfg["pretrained"]["phase_a_checkpoint"]),
                         map_location="cpu", weights_only=False)
    base = {k[len("transformer."):]: v
            for k, v in phase_a["model_state_dict"].items()
            if k.startswith("transformer.")}
    missing, _ = dit.load_state_dict(base, strict=False)
    style_missing = [k for k in missing if "style" in k]
    print(f"loaded phase A, {len(style_missing)} style keys will train from scratch")

    vocab = load_vocab(ROOT / "data/vocab_f5.txt")

    mel_spec = MelSpec(
        n_fft=model_cfg["mel"]["n_fft"], hop_length=model_cfg["mel"]["hop_length"],
        win_length=model_cfg["mel"]["win_length"],
        n_mel_channels=model_cfg["mel"]["n_mel_channels"],
        target_sample_rate=model_cfg["mel"]["target_sample_rate"],
        mel_spec_type=model_cfg["mel"]["mel_spec_type"],
    )
    cc = model_cfg["cfm"]
    cfm = CFMStyled(
        transformer=dit, sigma=cc["sigma"],
        audio_drop_prob=cc["audio_drop_prob"],
        cond_drop_prob=cc["cond_drop_prob"],
        style_drop_prob=cc["style_drop_prob"],
        mel_spec_module=mel_spec,
        frac_lengths_mask=tuple(cc["frac_lengths_mask"]),
        vocab_char_map=vocab,
    ).to(device)

    ema = EMA(cfm, beta=tc["ema_decay"])

    ds = SaturnDataset(dc["train_metadata"], dc["styles_path"], dc["data_root"])
    sampler = DynamicBatchSampler(ds, max_frames=tc["max_frames_per_batch"])
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn,
                        num_workers=4, pin_memory=True)

    # separate param groups for staged unfreezing
    style_params = [p for n, p in cfm.named_parameters() if "style" in n]
    base_params = [p for n, p in cfm.named_parameters() if "style" not in n]
    print(f"style: {sum(p.numel() for p in style_params)/1e6:.1f}M, "
          f"base: {sum(p.numel() for p in base_params)/1e6:.1f}M")

    opt = torch.optim.AdamW([
        {"params": style_params, "lr": tc["learning_rate"]},
        {"params": base_params, "lr": tc["learning_rate"]},
    ], weight_decay=tc["weight_decay"])

    scaler = torch.amp.GradScaler("cuda")
    save_dir = ROOT / "checkpoints/phase_b"
    save_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    max_steps = tc["max_steps"]
    accum = tc["grad_accumulation_steps"]
    freeze_until = tc["freeze_pretrained_steps"]
    running_loss = 0.0

    # freeze base params initially
    for p in base_params:
        p.requires_grad = False
    base_frozen = True

    print(f"phase B: {max_steps} steps, base frozen for {freeze_until}")

    cfm.train()
    while step < max_steps:
        for batch in loader:
            if base_frozen and step >= freeze_until:
                for p in base_params:
                    p.requires_grad = True
                base_frozen = False
                print(f"  step {step}: base params unfrozen")

            mel = batch["mel"].to(device)
            text = list_str_to_idx(batch["text"], vocab).to(device)
            style = batch["style"].to(device)
            lens = batch["mel_lens"].to(device)

            lr = cosine_lr(step, tc["warmup_steps"], max_steps, tc["learning_rate"])
            opt.param_groups[0]["lr"] = lr
            opt.param_groups[1]["lr"] = lr * (0.1 if not base_frozen else 0.0)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, _, _ = cfm(inp=mel, text=text, style=style, lens=lens)

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
                tag = " [frozen]" if base_frozen else ""
                print(f"step {step}/{max_steps}  loss={running_loss/tc['log_every']:.4f}  lr={lr:.2e}{tag}")
                running_loss = 0.0

            if step % tc["save_every"] == 0:
                ckpt = {"step": step, "model_state_dict": cfm.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "ema_state_dict": ema.state_dict(),
                        "model_config": model_cfg}
                torch.save(ckpt, save_dir / f"step_{step}.pt")
                torch.save(ckpt, save_dir / "latest.pt")
                for old in sorted(save_dir.glob("step_*.pt"),
                                  key=lambda p: int(p.stem.split("_")[1]))[:-tc["keep_last_n_checkpoints"]]:
                    old.unlink()

            if step >= max_steps:
                break

    torch.save({"step": step, "model_state_dict": cfm.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "model_config": model_cfg}, save_dir / "latest.pt")
    print(f"phase B done at step {step}")


if __name__ == "__main__":
    main()
