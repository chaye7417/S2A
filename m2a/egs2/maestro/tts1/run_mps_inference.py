#!/usr/bin/env python3
"""
MPS inference script for m2a (MIDI to Audio).
Converts a MIDI file to piano audio using the pre-trained model.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import pretty_midi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Model configuration
DEFAULT_CONFIG = "exp/tts_finetune_joint_transformer_hifigan_atepp_raw_proll/config.yaml"
DEFAULT_MODEL = "exp/tts_finetune_joint_transformer_hifigan_atepp_raw_proll/train.total_count.best.pth"


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def midi_to_pianoroll(midi_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Convert MIDI file to piano roll representation.

    Uses the same parameters as the original data_segments.py:
    - FRAME_SHIFT_MS = 12ms
    - up_sample_rate = FRAME_SHIFT_MS / 1000 * sample_rate = 192
    - frame_rate = sample_rate / up_sample_rate = 83.33 fps

    Args:
        midi_path: Path to MIDI file
        sample_rate: Audio sampling rate (default 16000)

    Returns:
        Piano roll array of shape (time_frames, 128)
    """
    # Parameters from original data_segments.py
    FRAME_SHIFT_MS = 12
    up_sample_rate = FRAME_SHIFT_MS / 1000 * sample_rate  # 192
    frame_rate = sample_rate / up_sample_rate  # 83.33 fps

    # Load MIDI
    pretty_midi.pretty_midi.MAX_TICK = 1e20
    midi = pretty_midi.PrettyMIDI(midi_path)

    # Get piano roll (128, time_frames)
    # Values are in 0-127 range (MIDI velocity)
    piano_roll = midi.get_piano_roll(fs=frame_rate)

    # Transpose to (time_frames, 128)
    piano_roll = piano_roll.T

    return piano_roll.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="MPS MIDI to Audio Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_mps_inference.py --midi input.mid --output output.wav
""",
    )
    parser.add_argument("--midi", type=str, required=True, help="Input MIDI file path")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV file path")
    # Keep legacy options for backwards compatibility
    parser.add_argument("--config", type=str, default=None, help="Override model config path")
    parser.add_argument("--model", type=str, default=None, help="Override model checkpoint path")
    args = parser.parse_args()

    # Get model paths
    config_path = args.config or DEFAULT_CONFIG
    model_path = args.model or DEFAULT_MODEL

    # Check files exist
    if not os.path.exists(args.midi):
        logger.error(f"MIDI file not found: {args.midi}")
        sys.exit(1)
    if not os.path.exists(config_path):
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    device = get_device()
    logger.info(f"Using device: {device}")

    # Load model using Text2Speech class
    logger.info("Loading model...")
    from espnet2.bin.gan_mta_inference import Text2Speech

    text2speech = Text2Speech(
        train_config=config_path,
        model_file=model_path,
        device=device,
        use_teacher_forcing=False,
    )
    logger.info("Model loaded successfully!")

    # Convert MIDI to piano roll
    logger.info(f"Processing MIDI: {args.midi}")
    piano_roll = midi_to_pianoroll(args.midi, sample_rate=16000)
    logger.info(f"Piano roll shape: {piano_roll.shape}")
    logger.info(f"Piano roll stats - min: {piano_roll.min():.1f}, max: {piano_roll.max():.1f}, non-zero: {np.count_nonzero(piano_roll)}")

    # Segment processing parameters (from original data_segments.py)
    num_segment_frame = 800  # Each segment is 800 frames

    # Run inference in segments
    logger.info("Running inference...")
    all_wavs = []

    num_segments = (piano_roll.shape[0] + num_segment_frame - 1) // num_segment_frame
    logger.info(f"Processing {num_segments} segments of {num_segment_frame} frames each")

    for seg_idx in range(num_segments):
        start_frame = seg_idx * num_segment_frame
        end_frame = min((seg_idx + 1) * num_segment_frame, piano_roll.shape[0])

        # Get segment
        segment = piano_roll[start_frame:end_frame]

        # Pad if last segment is shorter
        if segment.shape[0] < num_segment_frame:
            pad_length = num_segment_frame - segment.shape[0]
            segment = np.pad(segment, ((0, pad_length), (0, 0)), mode='constant')

        # Convert to tensor
        midi_tensor = torch.from_numpy(segment)

        # Run inference using Text2Speech
        output_dict = text2speech(midi=midi_tensor)

        # Extract audio
        if "wav" in output_dict:
            wav_segment = output_dict["wav"].cpu().numpy()
            all_wavs.append(wav_segment)
            logger.info(f"Segment {seg_idx + 1}/{num_segments}: shape={wav_segment.shape}, range=[{wav_segment.min():.4f}, {wav_segment.max():.4f}]")

    # Concatenate all segments
    if all_wavs:
        wav = np.concatenate(all_wavs, axis=0)
        logger.info(f"Generated audio shape: {wav.shape}")
        logger.info(f"Audio stats - min: {wav.min():.4f}, max: {wav.max():.4f}")

        # Output sample rate is 24000Hz (from config)
        output_fs = text2speech.fs if text2speech.fs else 24000
        sf.write(args.output, wav, output_fs, "PCM_16")
        logger.info(f"Audio saved to: {args.output} (sample rate: {output_fs}Hz)")
    else:
        logger.warning("No audio generated.")


if __name__ == "__main__":
    main()
