"""
Inference script for Spoken Language Understanding (SLU) models.

Supports multiple encoder architectures:
- CRDNN ASR Encoder (pre-trained on LibriSpeech) -> LSTM Encoder -> Attention GRU Decoder
- HuBERT Encoder -> Attention GRU Decoder

The model architecture is determined by the hyperparameters YAML file.

Usage:
    python inference_slu.py --wav_file <path_to_wav> --hparams <path_to_hparams> --device <cpu|cuda>
"""

import argparse
import os
import sys
import warnings

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

# Set environment variable and suppress warnings before importing SpeechBrain
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")


def load_audio(wav_file, target_sr=16000):
    """
    Load and preprocess audio file.
    """
    signal, sr = torchaudio.load(wav_file)

    # Resample if necessary
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        signal = resampler(signal)

    # Convert to mono if stereo
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    return signal.squeeze(0)


def get_model_type(hparams):
    """
    Determines the model type based on hyperparameters.
    """
    if "asr_model_path" in hparams:
        return "crdnn"
    elif "hubert_hub" in hparams or "hubert" in hparams:
        return "hubert"
    else:
        raise ValueError(
            "Could not determine model type from hyperparameters. "
            "Expected 'asr_model_path' for CRDNN or 'hubert_hub' for HuBERT."
        )


def run_inference_crdnn(hparams, wav_file, device="cpu"):
    """
    Run inference on a single wav file using CRDNN model.
    """
    from speechbrain.inference import EncoderDecoderASR

    # Load the tokenizer directly from file
    tokenizer = hparams["tokenizer"]
    tokenizer.load(hparams["tokenizer_file"])

    # Load the pretrained ASR encoder
    asr_model = EncoderDecoderASR.from_hparams(
        source=hparams["asr_model_path"],
        savedir=hparams["asr_model_path"],
    )
    asr_encoder = asr_model.mods.encoder  # type: ignore
    asr_encoder.eval()
    asr_encoder.to(device)

    # Load the SLU model
    modules = hparams["modules"]
    for module in modules.values():
        module.to(device)
        module.eval()

    # Load the checkpointer
    checkpointer = hparams["checkpointer"]
    checkpointer.recover_if_possible()

    # Load and preprocess audio
    print(f"Loading audio from: {wav_file}")
    signal = load_audio(wav_file, target_sr=hparams["sample_rate"])
    signal = signal.unsqueeze(0).to(device)  # Add batch dimension

    # Calculate actual length in samples
    signal_length = torch.tensor([signal.shape[1]]).to(device)
    # Normalize to relative length (0-1)
    rel_length = signal_length / signal_length.float()

    # Run through ASR encoder
    print("Encoding audio...")
    with torch.no_grad():
        # Pass signal with lengths parameter
        asr_encoded = asr_encoder(signal, lengths=rel_length)

        # Run through SLU encoder
        encoded = hparams["slu_enc"](asr_encoded)

        # Decode using beam search
        print("Decoding semantic tokens...")
        hyps, _, _, _ = hparams["beam_searcher"](encoded, rel_length)

        # Get the best hypothesis
        predicted_tokens = hyps[0]

        # Decode tokens to text
        predicted_semantics = tokenizer.decode_ids(predicted_tokens)

    return predicted_semantics


def run_inference_hubert(hparams, wav_file, device="cpu"):
    """
    Run inference on a single wav file using HuBERT model.
    """
    # Load the tokenizer directly from file
    tokenizer = hparams["tokenizer"]
    tokenizer.load(hparams["tokenizer_file"])

    # Move HuBERT to device
    hubert = hparams["hubert"]
    hubert.to(device)
    hubert.eval()

    # Load the SLU model modules
    modules = hparams["modules"]
    for module in modules.values():
        module.to(device)
        module.eval()

    # Load the checkpointer
    checkpointer = hparams["checkpointer"]
    checkpointer.recover_if_possible()

    # Load and preprocess audio
    print(f"Loading audio from: {wav_file}")
    signal = load_audio(wav_file, target_sr=hparams["sample_rate"])
    signal = signal.unsqueeze(0).to(device)  # Add batch dimension

    # Calculate actual length in samples
    signal_length = torch.tensor([signal.shape[1]]).to(device)
    # Normalize to relative length (0-1)
    rel_length = signal_length / signal_length.float()

    # Run through HuBERT encoder
    print("Encoding audio...")
    with torch.no_grad():
        hubert_out = hubert(signal, rel_length)

        # Decode using beam search
        print("Decoding semantic tokens...")
        hyps, _, _, _ = hparams["beam_searcher"](hubert_out, rel_length)

        # Get the best hypothesis
        predicted_tokens = hyps[0]

        # Decode tokens to text
        predicted_semantics = tokenizer.decode_ids(predicted_tokens)

    return predicted_semantics


def main():
    parser = argparse.ArgumentParser(
        description="Unified SLU Inference Script for CRDNN and HuBERT models"
    )
    parser.add_argument(
        "--wav_file", required=True, type=str, help="Path to the input wav file"
    )
    parser.add_argument(
        "--hparams",
        required=True,
        type=str,
        help="Path to hyperparameters file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on",
    )

    args = parser.parse_args()

    # Load hyperparameters
    print(f"Loading hyperparameters from: {args.hparams}")
    with open(args.hparams) as f:
        hparams = load_hyperpyyaml(f)

    # Check if GPU is available
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        device = "cpu"

    print(f"Using device: {device}")

    # Determine model type
    model_type = get_model_type(hparams)
    print(f"Detected model type: {model_type}")

    # Run inference based on model type
    try:
        if model_type == "crdnn":
            predicted_semantics = run_inference_crdnn(hparams, args.wav_file, device)
        else:  # hubert
            predicted_semantics = run_inference_hubert(hparams, args.wav_file, device)

        print("\n" + "=" * 50)
        print("INFERENCE RESULTS")
        print("=" * 50)
        print(f"Input file: {args.wav_file}")
        print(f"Predicted semantics: {predicted_semantics}")
        print("=" * 50)

    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
