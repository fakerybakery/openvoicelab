# OpenVoiceLab

[**Documentation**](https://openvoicelab.github.io/) | [**Join Discord**](https://discord.gg/7C2PPhgtkf)

Run, compare and fine-tune the latest open text-to-speech models from one web UI.

- **Generate** speech with 10 model families: preset voices, zero-shot voice cloning, voice design from a text description, style instructions and multi-speaker dialogue.
- **Voice library**: save reference clips once and reuse them with any model. Transcripts are filled in automatically with Whisper.
- **Models** install with one click into their own isolated environments, so models with conflicting dependencies (different `transformers` or `torch` versions) work side by side.
- **Datasets**: turn long recordings into a training dataset (VAD + Whisper segmentation).
- **Fine-tune** VibeVoice with LoRA and load the adapter for generation.

> [!NOTE]
> OpenVoiceLab is in beta. Feedback and contributions are welcome!

## Supported models

| Model | Checkpoints | Cloning | Presets | Voice design | Multi-speaker | Languages | License |
| --- | --- | :-: | :-: | :-: | :-: | --- | --- |
| [VibeVoice](https://github.com/vibevoice-community/VibeVoice) | 1.5B, 7B, Realtime 0.5B | ✓ | Realtime | | 4 | English, Chinese | MIT |
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | 0.6B / 1.7B CustomVoice, VoiceDesign, Base | Base | CustomVoice | VoiceDesign | 4 | 10 | Apache-2.0 |
| [VoxCPM](https://github.com/OpenBMB/VoxCPM) | VoxCPM2, 1.5, 0.5B | ✓ | | VoxCPM2 | | 30 | Apache-2.0 |
| [Chatterbox](https://github.com/resemble-ai/chatterbox) | Turbo, Multilingual V3/V2, English, Nano | ✓ | | | 4 | 23 | MIT |
| [OmniVoice](https://github.com/k2-fsa/OmniVoice) | OmniVoice | ✓ | | ✓ | | 600+ | ⚠️ CC-BY-NC-4.0 |
| [Echo-TTS](https://github.com/jordandare/echo-tts) | Base | ✓ | | | | English | ⚠️ CC-BY-NC-SA-4.0 |
| [MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) | TTS v1.5 (8B), Local v1.5 (4B), TTSD, VoiceGenerator, Nano | ✓ | | VoiceGenerator | TTSD (5) | 31 | Apache-2.0 |
| [Kitten TTS](https://github.com/KittenML/KittenTTS) | Mini, Micro, Nano (0.8) | | ✓ | | | English | Apache-2.0 |
| [Pocket TTS](https://huggingface.co/kyutai/pocket-tts) | English, French, German, Spanish, Italian, Portuguese, Dutch (110M and 24L 340M) | ✓ | ✓ | | 4 | 7 | CC-BY-4.0 |
| [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) | 82M | | ✓ | | | 7 | Apache-2.0 |

⚠️ = non-commercial license. Check each model's license before using its output commercially. Echo-TTS's license also applies to the audio it generates.

Multi-speaker support for models without native dialogue (Qwen3-TTS, Chatterbox, Pocket TTS) works line by line: OpenVoiceLab generates each line with that speaker's voice and joins them.

**Hardware:** everything runs on NVIDIA GPUs. Most models also run on Apple Silicon (MPS) and CPU; Kitten TTS, Pocket TTS and Kokoro are fast on CPU. Fine-tuning needs an NVIDIA GPU (16 GB+ VRAM for VibeVoice 1.5B).

## Quick start

You need Python 3.10+ (or [uv](https://docs.astral.sh/uv/), which installs Python for you) and git.

**Linux / macOS**

```bash
git clone https://github.com/fakerybakery/openvoicelab && cd openvoicelab
./scripts/setup.sh
./scripts/run.sh
```

**Windows**

```cmd
git clone https://github.com/fakerybakery/openvoicelab && cd openvoicelab
scripts\setup.bat
scripts\run.bat
```

Then open http://localhost:7860. Choose a model on the **Generate** tab and click **Load model**. Its environment is installed the first time you load it (you can also install models from the **Models** tab), and its weights are downloaded from Hugging Face.

To update: `./scripts/update.sh` (or `scripts\update.bat`).

### Manual setup

```bash
uv venv --python 3.12 venv
uv pip install --python venv/bin/python --torch-backend=auto -e .
venv/bin/openvoicelab
```

## Command line

```bash
openvoicelab                      # launch the web UI (same as `openvoicelab ui`)
openvoicelab ui --port 7861 --share
openvoicelab models               # list models and whether they're installed
openvoicelab install kokoro qwen3_tts   # or `install all`

# Preset voice
openvoicelab tts kokoro "Hello there" --voice af_heart -o hello.wav

# Clone a voice (some models want the clip's transcript too)
openvoicelab tts qwen3_tts "Hello there" --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --ref my_voice.wav --ref-text "What I say in the clip." -o clone.wav

# Describe a voice
openvoicelab tts voxcpm "Hello there" --design "A warm, older British man" -o design.wav

# Dialogue: one --ref (or --voice) per speaker
openvoicelab tts vibevoice "$(cat examples/multi_speaker_example.txt)" \
    --ref ovl/voices/Alice.wav --ref ovl/voices/Carter.wav -o podcast.wav

# Model-specific parameters
openvoicelab tts echo_tts "Hello there" --ref my_voice.wav --param num_steps=20 --seed 42
```

## Multi-speaker scripts

Write one line per turn with a speaker label. Labels may start at 0 or 1; speakers are numbered in order of first appearance and get voices 1, 2, 3...

```
Speaker 1: Hello everyone! Welcome to the show.
Speaker 2: Thanks for having me!
Speaker 1: Let's dive right in.
```

A line without a label continues with the previous speaker; a script with no labels at all rotates through the speakers line by line. See `examples/` for longer scripts.

## Where things are stored

Everything OpenVoiceLab writes goes in one folder: the directory you run it from, or `OVL_HOME` if that's set.

| Folder | Contents |
| --- | --- |
| `envs/` | One Python environment per installed model (remove from the Models tab) |
| `voices/` | Your voice library: audio clips plus `.txt` transcripts |
| `outputs/` | Generated audio |
| `data/` | Datasets built on the Data tab |
| `training_runs/` | Fine-tuning runs and LoRA adapters |
| `logs/` | App log |

Model weights go in the regular Hugging Face cache (`~/.cache/huggingface`). For gated models, log in with `hf auth login`.

## How it works

Each model family is described by a small spec in `ovl/backends/catalog/` (checkpoints, capabilities, parameters, pinned requirements) and implemented in `ovl/backends/impl/`. When you load a model, OpenVoiceLab starts a worker process in that model's environment (`python -m ovl.runtime.worker`), which speaks JSON lines over stdin/stdout. Unloading a model ends the worker, so all of its memory is freed. A crash in a model doesn't take down the UI either.

To add a model, add a catalog spec and a `Backend` class implementing `load()` and `generate()`. See `ovl/backends/impl/kokoro.py` for a minimal example.

## Troubleshooting

- **A model fails to load**: open **Worker log** on the Generate tab for the full error. Reinstalling the model's environment from the Models tab fixes most broken installs.
- **Out of memory**: unload the model, then pick a smaller checkpoint, or use `cpu` as the device.
- **CUDA not used**: run `scripts/update.sh` to reinstall PyTorch with `--torch-backend=auto`, then reinstall the model from the Models tab.
- **Out of memory while training**: lower the batch size or turn on gradient checkpointing.

## Acknowledgements

- [ai-voice-cloning](https://git.ecker.tech/ecker/ai-voice-cloning), which inspired this project
- The teams behind every model listed above
- Thanks to chocoboy for early feedback on Windows

## License

OpenVoiceLab is licensed under the BSD-3-Clause license. See [LICENSE](LICENSE). Models you install have their own licenses (see the table above).
