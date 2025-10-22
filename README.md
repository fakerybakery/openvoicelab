# OpenVoiceLab (Beta)

[**Documentation**](https://openvoicelab.github.io/) | [**Join Discord**](https://discord.gg/7C2PPhgtkf)

> [!IMPORTANT]
> OpenVoiceLab is currently in beta. Some things still need to be improved - especially in the finetuning process. Windows support is still WIP. Feedback and contributions are welcome!

A beginner-friendly interface for finetuning and running text-to-speech models. Currently supports VibeVoice.

## What is this?

OpenVoiceLab provides a simple web interface for working with the VibeVoice text-to-speech model. You can:

- **Finetune models** on your own voice data to create custom voices
- **Generate speech** from text using pretrained or finetuned models
- **Experiment** with models through an easy-to-use web UI

The goal is to make state-of-the-art voice synthesis accessible to anyone interested in exploring TTS technology, whether you're a developer, researcher, content creator, or hobbyist.

## Requirements

Before you begin, make sure you have:

- **Python 3.9 or newer** - Check your version with `python3 --version`
- **CUDA-compatible NVIDIA GPU** - At least 16 GB of VRAM is recommended for training the 1.5B parameter model
  - For inference (generating speech), you can get by with less VRAM or even CPU-only mode, though it will be slower
- **Operating System** - Linux, macOS, or Windows

## Quick Start

The easiest way to get started is using the provided setup scripts. These will create a Python virtual environment, install all dependencies, and launch the interface.

### Linux/macOS

```bash
./scripts/setup.sh
./scripts/run.sh
```

### Windows

```cmd
scripts\setup.bat
scripts\run.bat
```

After running these commands, the web interface will launch automatically. Open your browser and navigate to:

```
http://localhost:7860
```

If the browser doesn't open automatically, you can manually visit this address.

## Manual Setup

If you prefer to set things up yourself, or if the scripts don't work on your system:

1. **Create a virtual environment** (recommended to avoid conflicts with other Python packages):
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the interface**:
   ```bash
   python -m ovl.cli
   ```

   Then open your browser to `http://localhost:7860`

## What's Next?

Once you have OpenVoiceLab running, you can:

- Start with inference to generate speech from a pretrained model
- Prepare your own voice dataset for finetuning
- Experiment with different model parameters and training configurations

Detailed usage instructions are available in the interface itself.

## Usage

### Single Speaker Inference

Simply enter your text in the text box and select a voice for Speaker 1. The model will generate speech in that voice.

### Multi-Speaker Inference (Up to 4 Speakers)

OpenVoiceLab supports multi-speaker conversations with up to 4 distinct voices. The implementation follows the VibeVoice gradio_demo.py approach with 0-indexed speakers.

**Two ways to use multi-speaker:**

1. **Format your text with speaker labels** (0-indexed):
```
Speaker 0: Hello there! How are you doing today?
Speaker 1: I'm doing great, thanks for asking!
Speaker 0: That's wonderful to hear.
Speaker 2: Hey everyone, mind if I join the conversation?
Speaker 3: Of course not! The more the merrier.
```

2. **Enter plain text** and speakers will be auto-assigned in rotation:
```
Hello there! How are you doing today?
I'm doing great, thanks for asking!
That's wonderful to hear.
```

**Instructions:**
1. Set the "Number of Speakers" slider (1-4)
2. Select a voice for each speaker (Speaker 0, Speaker 1, Speaker 2, Speaker 3)
3. Enter your text (with or without `Speaker X:` labels)
4. If you don't use speaker labels, text will be auto-assigned to speakers in rotation

**Tips:**
- Speakers are 0-indexed: Speaker 0, Speaker 1, Speaker 2, Speaker 3 (matching VibeVoice's format)
- Only the number of speakers you select will be shown in the UI
- Voice samples are passed to the model in the same order as selected
- The voices are stored in the `voices/` folder - you can add your own reference audio files there
- Example scripts are available in the `examples/` folder:
  - `multi_speaker_example.txt` - A 2-speaker conversation
  - `four_speaker_example.txt` - A 4-speaker discussion

## Troubleshooting

**Out of memory errors during training**: Try reducing the batch size or using a smaller model variant.

**CUDA not available**: Make sure you have NVIDIA drivers and PyTorch with CUDA support installed. The setup scripts should handle this automatically.

**Import errors**: Ensure you've activated the virtual environment before running the CLI.

## Acknowledgements

- [ai-voice-cloning](https://git.ecker.tech/ecker/ai-voice-cloning): This project is heavily inspired by the 2023-era AI Voice Cloning project
- [VibeVoice](https://github.com/microsoft/VibeVoice): This project is essentially a wrapper for VibeVoice, kudos to the Microsoft team behind VibeVoice

- Thanks to chocoboy for providing initial feedback on Windows

## Roadmap

- Support for auto-updating OpenVoiceLab GUI

## License

OpenVoiceLab is licensed under the BSD-3-Clause license. See the [LICENSE](LICENSE) file for details.
