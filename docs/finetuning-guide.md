# Finetuning Guide

This guide walks you through finetuning VibeVoice on your own voice data to create custom voices.

## Overview

Finetuning allows you to adapt the VibeVoice model to better replicate a specific voice. The process involves:

1. Preparing audio data
2. Processing the data into a training dataset
3. Training the model with LoRA (Low-Rank Adaptation)
4. Using the finetuned model for inference

## Prerequisites

- A folder containing audio files (MP3, WAV, FLAC, or M4A format)
- At least 30 minutes of clean audio for decent results (more is better)
- 16+ GB VRAM for training the 1.5B model
- OpenVoiceLab installed and running

## Step 1: Prepare Your Audio Files

### What makes good training data?

- **Clean audio**: Minimal background noise, no music, no overlapping voices
- **Single speaker**: All audio shoul d be from the same person
- **Natural speech**: Conversational or read-aloud content works best
- **Variety**: Different sentences, emotions, and speaking styles help
- **Length**: Individual files can be any length - they will be automatically segmented

### Organizing your files

Simply place all your audio files in a single folder. The format doesn't matter much:

```
my_voice_data/
├── recording1.mp3
├── recording2.wav
├── podcast_clip.m4a
└── audiobook_sample.flac
```

You don't need to segment the audio yourself or provide transcriptions - OpenVoiceLab handles this automatically.

## Step 2: Create a Training Dataset

Open OpenVoiceLab in your browser (default: `http://localhost:7860`) and navigate to the **Data** tab.

### Configuration

1. **Input Directory**: Enter the full path to your folder containing audio files
   ```
   /path/to/my_voice_data
   ```

2. **Dataset Name**: Give your dataset a memorable name (e.g., `john_voice`, `my_podcast_voice`)
   - Use letters, numbers, and underscores only
   - This name will be used to reference the dataset later

3. **Whisper Model**: Choose a transcription model
   - `openai/whisper-base` - Good balance of speed and accuracy (recommended for most users)
   - `openai/whisper-small` - Better accuracy, slightly slower
   - `openai/whisper-medium` - Even better accuracy, noticeably slower
   - `openai/whisper-large-v3-turbo` - Best accuracy, requires more VRAM and time

### Processing

Click **Start Processing**. The system will:

1. **Segment audio** using Silero VAD (Voice Activity Detection)
   - Automatically splits files into speech segments
   - Removes silence and non-speech audio
   - Each segment becomes a training sample

2. **Transcribe segments** using Whisper
   - Generates text transcriptions for each segment
   - No manual transcription needed

3. **Save in LJSpeech format** at `data/{dataset_name}/`
   - `wavs/` folder contains all audio segments
   - `metadata.csv` contains transcriptions in format: `filename|text|text`
   - `info.json` contains dataset metadata

Processing time depends on:
- Amount of audio
- Whisper model size
- Your hardware (CPU/GPU)

A typical 30-minute audio set takes 3-10 minutes to process on a modern consumer GPU.

### Verify your dataset

After processing completes, you'll see your dataset listed with:
- Number of samples created
- Creation timestamp
- Storage location

You can inspect the dataset manually:
```bash
ls data/my_dataset_name/wavs/  # View audio segments
head data/my_dataset_name/metadata.csv  # View transcriptions
```

## Step 3: Train the Model

Navigate to the **Training** tab.

### Configuration

1. **Dataset**: Select your dataset from the dropdown
   - If you don't see it, click "Refresh Datasets"

2. **Model Path**:
   - Default: `vibevoice/VibeVoice-1.5B` (recommended, but use 7B if you have enough VRAM)
   - You can also use a local path if you've downloaded the model

3. **Training Parameters** (click "Training Parameters" accordion to expand):

   - **Epochs**: Number of times to train on the full dataset
     - Start with 3-5 epochs
     - More epochs = longer training but potentially better results
     - Too many epochs can lead to overfitting

   - **Batch Size**: Number of samples processed together
     - Default: 4 (works on 16GB VRAM)
     - Reduce to 2 or 1 if you get out-of-memory errors
     - Increase to 8 if you have 24GB+ VRAM

   - **Learning Rate**: How quickly the model adapts
     - Default: 1e-4 (0.0001) works well for most cases
     - Don't change unless you know what you're doing

   - **LoRA Rank (r)**: Controls adapter complexity
     - Default: 8 (good balance)
     - Higher values (16, 32) = more parameters = better quality but slower and more VRAM
     - Lower values (4) = faster but may sacrifice quality

### Start Training

Click **Start Training**. The system will:

1. Load the base VibeVoice model
2. Add LoRA adapters (lightweight finetuning approach)
3. Train on your dataset
4. Save checkpoints to `training_runs/output_{dataset_name}/`
5. Launch TensorBoard for monitoring

### Monitor Progress

**TensorBoard** (embedded in the UI):
- Loss curves show training progress
- Click "Refresh TensorBoard" after starting training (wait 10-20 seconds for it to initialize)

**Training Logs** (expandable accordion):
- Real-time training output
- Shows current epoch, step, loss values
- Useful for debugging if something goes wrong

**Training History**:
- Lists all training runs
- Shows status (running/completed)
- Click "Refresh Runs" to update

### Training Time

Training time varies based on:
- Dataset size (number of samples)
- Number of epochs
- Batch size
- Hardware

You can check training progress and the ETA at any time by opening the training logs.

### When to stop

The training will complete automatically after the specified number of epochs. However, you can:
- Click **Stop Training** at any time
- The latest checkpoint will still be usable

## Step 4: Use Your Finetuned Model

Navigate to the **Inference** tab.

### Loading the finetuned model

1. **Model Path**: Keep the default `vibevoice/VibeVoice-1.5B`

2. **Load LoRA Adapter**: Check this box

3. **LoRA Path**: Enter the path to your training output
   ```
   training_runs/output_{dataset_name}
   ```
   Replace `{dataset_name}` with your actual dataset name.

4. Click **Load Model**

The model will load the base VibeVoice weights plus your custom LoRA adapter.

### Generate speech

1. Enter text in the **Text** box
2. **Enable Voice Cloning**:
   - Check this to use a reference voice from the `voices/` folder
   - Uncheck to use the finetuned model's learned voice directly
3. **CFG Scale**: Controls how closely the model follows the reference voice
   - Range: 1.0 - 2.0
   - Higher = more faithful to reference, but may sound less natural
   - Default: 1.3 is usually good
4. Click **Generate Speech**

Generated audio will be saved to `outputs/` and played in the browser.

## Tips for Best Results

### Data quality matters most
- Use the cleanest audio you can find
- Remove sections with background noise, music, or other speakers
- 30-60 minutes of good data beats hours of noisy data

### Monitor for overfitting
- If the model sounds robotic or loses naturalness, you may have overtrained
- Try reducing epochs (e.g., 3 instead of 5)
- Try reducing LoRA rank

### Experiment with parameters
- Start with defaults
- If results aren't good enough, try:
  - More data
  - More epochs (4-5)
  - Higher LoRA rank (16)

### Hardware limitations
If you run out of VRAM:
- Reduce batch size to 2 or 1
- Use a smaller LoRA rank (4 instead of 8)
- Train on fewer samples at a time

## Troubleshooting

### "No audio segments found after VAD processing"
- Your audio may be too quiet or too noisy
- Try using different audio files with clearer speech
- Check that your audio files are valid and not corrupted

### "Out of memory" during training
- Reduce batch size
- Reduce LoRA rank
- Close other GPU-using applications
- Use a smaller dataset

### Training loss not decreasing
- Check that your dataset has enough variety
- Try increasing learning rate slightly (1.5e-4)
- Make sure your audio quality is good

### Generated audio sounds nothing like the target voice
- Finetune for more epochs
- Increase LoRA rank for more model capacity
- Check your training data quality
- Make sure you're loading the LoRA adapter correctly in inference

### Model takes forever to generate
- This is normal for this type of model
- The 7B model generates roughly 2 seconds of audio per second of compute time on a RTX 4090
- Use a GPU (CUDA or MPS) instead of CPU for much faster generation

## Next Steps

- Try different training configurations to see what works best for your voice
- Experiment with different reference voices in the `voices/` folder
- Share your results and get help in the community forums
