# SaturnTTS

Emotional text-to-speech built on [F5-TTS](https://github.com/SWivid/F5-TTS) with per-block style cross-attention and multi-term classifier-free guidance. Fine-tuned on the [Expresso](https://speechbot.github.io/expresso/) dataset (11h, 4 speakers, 9 expressive styles).

The model supports zero-shot voice cloning with independent control over speaker identity and emotional style -- you can make any voice sound happy, sad, confused, whispery, etc. by adjusting the style guidance scale.

## How it works

SaturnTTS extends F5-TTS's diffusion transformer (DiT) by inserting a cross-attention layer before each of the 22 transformer blocks. These layers attend from the main audio stream to a learned style embedding, letting the model condition its generation on discrete emotion labels without modifying the pretrained weights.

Training uses three-way CFG dropout so that inference can independently control overall conditioning strength and emotional intensity:

```
v_guided = v_full + s1 * (v_full - v_null) + s2 * (v_full - v_no_style)
```

Where `s1` controls adherence to the reference speaker and `s2` dials emotional expressiveness.

## Quickstart

```bash
git clone https://github.com/josealarconsilva/SaturnTTS.git
cd SaturnTTS && pip install -e .
```

Download the pretrained checkpoint from [HuggingFace](https://huggingface.co/JoseAlarcon/saturn-tts):

```bash
huggingface-cli download JoseAlarcon/saturn-tts --local-dir checkpoints/
```

Generate speech:

```python
from saturn_tts.infer.generate import load_model, load_vocoder, generate

model, cfg = load_model("checkpoints/step_15000.pt")
vocoder = load_vocoder()

wav = generate(
    model, vocoder,
    text="This is absolutely incredible news!",
    ref_audio_path="ref.wav",
    ref_text="The transcript of your reference audio.",
    style="happy",
    styles_map={"happy": 5, "sad": 7, "default": 1, ...},
    vocab_char_map=vocab,           # F5-TTS Emilia vocab
    style_cfg_strength=5.0,         # higher = more emotional
)
```

## Chat demo

The repo includes a Gradio chat app that pairs SaturnTTS with Gemma 4 and Whisper for a fully voice-enabled AI conversation:

```bash
pip install -e ".[chat]"
python scripts/chat_server.py
```

The LLM automatically tags each response with an emotion (`[happy]`, `[sad]`, `[confused]`, etc.) and the TTS renders it with the appropriate style. No manual controls needed.

## Training

### Data preparation

Download the [Expresso dataset](https://huggingface.co/datasets/ylacombe/expresso) and run:

```bash
python scripts/prepare_data.py
```

### Phase A: domain adaptation

Fine-tunes the pretrained F5-TTS on Expresso without style conditioning. This adapts the acoustic model to the new speakers. Usually converges in ~5K steps.

```bash
python scripts/train_phase_a.py
```

### Phase B: style conditioning

Loads the Phase A checkpoint, adds style cross-attention blocks (39M new params), and trains with multi-term CFG dropout. The base DiT is frozen for the first 10K steps, then unfrozen with a 10x lower learning rate.

```bash
python scripts/train_phase_b.py
```

## Architecture

| Component | Params | Source |
|-----------|--------|--------|
| DiT backbone (dim=1024, depth=22) | 336M | Pretrained F5-TTS |
| Style embedding + 22x cross-attention | 39M | Trained from scratch |
| Vocos vocoder | 14M | Pretrained (frozen) |

## Available styles

`default`, `confused`, `emphasis`, `enunciated`, `essentials`, `happy`, `laughing`, `sad`, `whisper`

## License

CC BY-NC 4.0 (matching F5-TTS and Expresso).

## Acknowledgments

- [F5-TTS](https://github.com/SWivid/F5-TTS) for the pretrained backbone
- [Expresso](https://speechbot.github.io/expresso/) for the expressive speech dataset
- [Vocos](https://github.com/gemelo-ai/vocos) for the vocoder
- The multi-term CFG approach draws from [F5-TTS-Emotional-CFG](https://github.com/RaduBolbo/F5-TTS-Emotional-CFG)
