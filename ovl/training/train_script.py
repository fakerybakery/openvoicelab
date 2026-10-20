"""Kept for backwards compatibility; training now runs `python -m vibevoice.finetune.train_vibevoice`
inside the VibeVoice environment (see trainer.py)."""

from vibevoice.finetune.train_vibevoice import main

if __name__ == "__main__":
    main()
