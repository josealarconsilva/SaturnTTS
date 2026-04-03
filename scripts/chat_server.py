#!/usr/bin/env python3
"""
Gradio chat app: speak or type -> Gemma 4 responds -> SaturnTTS speaks back.

The LLM tags each response with an emotion like [happy] or [sad], and the
TTS engine uses that to modulate the voice. No manual controls needed.
"""

import sys
sys.path.insert(0, "/home/jose/SaturnTTS/src")

import json
import re
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer

from saturn_tts.infer.generate import (
    load_model as load_tts, load_vocoder, generate as tts_generate,
)

ROOT = Path("/home/jose/SaturnTTS")
TMP = Path(tempfile.mkdtemp(prefix="saturn_chat_"))

REF_AUDIO = ROOT / "data" / "shadowheart_ref.wav"
REF_TEXT = "You Selunites are coddled. Decadent. Drunk on hope."

EMOTIONS = {"default", "happy", "sad", "confused", "whisper", "laughing"}

SYSTEM = (
    "You are a witty, sharp-tongued companion. Keep it to 1-3 sentences.\n\n"
    "Tag every response with one of these emotions in brackets: "
    "[default], [happy], [sad], [confused], [whisper], [laughing].\n"
    "Pick whichever fits your tone. Examples:\n"
    '- "[happy] Oh, brilliant! I knew you had it in you."\n'
    '- "[confused] Wait... you want me to do what, exactly?"\n'
    '- "[whisper] Just between us, I think something is off."'
)


def load_models():
    device = torch.device("cuda")

    print("  tts...", flush=True)
    tts, _ = load_tts(ROOT / "checkpoints/phase_b/step_15000.pt", device)
    vocoder = load_vocoder(device)

    print("  whisper...", flush=True)
    stt = whisper.load_model("base", device=device)

    print("  gemma...", flush=True)
    gemma_path = ROOT / "pretrained/gemma-4-e4b-it"
    if not gemma_path.exists():
        gemma_path = "google/gemma-4-e4b-it"
    tokenizer = AutoTokenizer.from_pretrained(str(gemma_path))
    llm = AutoModelForCausalLM.from_pretrained(
        str(gemma_path), dtype=torch.bfloat16, device_map="cuda",
    )

    with open(ROOT / "data/styles.json") as f:
        styles = json.load(f)
    with open(ROOT / "data/vocab_f5.txt") as f:
        vocab = {line.rstrip("\n"): i for i, line in enumerate(f)}

    print(f"  ready, peak vram: {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)

    return dict(tts=tts, vocoder=vocoder, stt=stt, llm=llm,
                tokenizer=tokenizer, styles=styles, vocab=vocab, device=device)


def transcribe(m, audio_path):
    if not audio_path:
        return ""
    return m["stt"].transcribe(audio_path, language="en")["text"].strip()


def parse_emotion(text):
    """Pull out [emotion] tag, return (emotion, clean_text)."""
    match = re.match(r"\[(\w+)\]\s*(.*)", text, re.DOTALL)
    if match and match.group(1).lower() in EMOTIONS:
        return match.group(1).lower(), match.group(2).strip()
    return "default", text.strip()


def ask_llm(m, user_text, history):
    messages = [
        {"role": "user", "content": SYSTEM},
        {"role": "assistant", "content": "[default] Got it. I'll tag my emotions."},
    ]
    for turn in history[-10:]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["raw"]})
    messages.append({"role": "user", "content": user_text})

    inputs = m["tokenizer"].apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
    )
    ids = inputs["input_ids"].to(m["device"])
    mask = inputs["attention_mask"].to(m["device"])

    with torch.no_grad():
        out = m["llm"].generate(input_ids=ids, attention_mask=mask,
                                max_new_tokens=200, temperature=0.7,
                                top_p=0.9, do_sample=True)

    return m["tokenizer"].decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def speak(m, text, emotion):
    wav = tts_generate(
        m["tts"], m["vocoder"], text=text, ref_audio_path=REF_AUDIO,
        ref_text=REF_TEXT, style=emotion, styles_map=m["styles"],
        vocab_char_map=m["vocab"], cfg_strength=2.0,
        style_cfg_strength=4.0, steps=32, device=m["device"],
    )
    audio = wav.cpu().squeeze().numpy()
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio *= 0.7 / peak

    path = TMP / f"r_{int(time.time()*1000)}.wav"
    sf.write(str(path), audio, 24000)
    return str(path)


def handle_turn(audio_in, text_in, chatbot, hist):
    m = handle_turn._m

    user = transcribe(m, audio_in) if audio_in else (text_in or "")
    if not user.strip():
        return chatbot, None, hist, "", None

    raw = ask_llm(m, user, hist)
    emotion, clean = parse_emotion(raw)
    audio_path = speak(m, clean, emotion)

    hist.append({"user": user, "raw": raw})
    chatbot.append({"role": "user", "content": user})
    chatbot.append({"role": "assistant", "content": f"*[{emotion}]* {clean}"})

    return chatbot, audio_path, hist, "", None


def ui():
    with gr.Blocks(title="SaturnChat") as app:
        gr.Markdown("# SaturnChat\n*speak or type -- AI responds with voice and emotion*")
        hist = gr.State([])
        chat = gr.Chatbot(height=500)
        audio_out = gr.Audio(type="filepath", autoplay=True)

        with gr.Row():
            txt = gr.Textbox(placeholder="type a message...", show_label=False, scale=5)
            btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            mic = gr.Audio(sources=["microphone"], type="filepath", label="or speak")
            clr = gr.Button("Clear")

        io = dict(inputs=[mic, txt, chat, hist],
                  outputs=[chat, audio_out, hist, txt, mic])
        btn.click(handle_turn, **io)
        txt.submit(handle_turn, **io)
        clr.click(lambda: ([], None, [], "", None), inputs=[], outputs=list(io["outputs"]))

    return app


if __name__ == "__main__":
    print("loading models...", flush=True)
    handle_turn._m = load_models()
    app = ui()
    print("starting server...", flush=True)
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
