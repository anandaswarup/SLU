"""
Evaluation script for Spoken Language Understanding (SLU) models.

Supports multiple encoder architectures:
- CRDNN ASR Encoder (pre-trained on LibriSpeech) -> LSTM Encoder -> Attention GRU Decoder
- HuBERT Encoder -> Attention GRU Decoder

The model architecture is determined by the hyperparameters YAML file.

Usage:
    python evaluate_slu.py --test_csv <path_to_test.csv> --output_csv <path_to_results.csv> \
        --hparams <path_to_hparams> --device <cpu|cuda>
"""

import argparse
import csv
import os
import sys
import warnings

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

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


def setup_crdnn_model(hparams, device="cpu"):
    """
    Setup and load CRDNN model components.
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

    return tokenizer, {"asr_encoder": asr_encoder}


def setup_hubert_model(hparams, device="cpu"):
    """
    Setup and load HuBERT model components.
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

    return tokenizer, {"hubert": hubert}


def run_inference_crdnn(hparams, encoder_components, tokenizer, wav_file, device="cpu"):
    """
    Run inference on a single wav file using CRDNN model.
    """
    asr_encoder = encoder_components["asr_encoder"]

    try:
        # Load and preprocess audio
        signal = load_audio(wav_file, target_sr=hparams["sample_rate"])
        signal = signal.unsqueeze(0).to(device)  # Add batch dimension

        # Calculate actual length in samples
        signal_length = torch.tensor([signal.shape[1]]).to(device)
        # Normalize to relative length (0-1)
        rel_length = signal_length / signal_length.float()

        # Run through ASR encoder
        with torch.no_grad():
            # Pass signal with lengths parameter
            asr_encoded = asr_encoder(signal, lengths=rel_length)

            # Run through SLU encoder
            encoded = hparams["slu_enc"](asr_encoded)

            # Decode using beam search
            hyps, _, _, _ = hparams["beam_searcher"](encoded, rel_length)

            # Get the best hypothesis
            predicted_tokens = hyps[0]

            # Decode tokens to text
            predicted_semantics = tokenizer.decode_ids(predicted_tokens)

        return predicted_semantics

    except Exception as e:
        print(f"Error processing {wav_file}: {e}", file=sys.stderr)
        return f"ERROR: {str(e)}"


def run_inference_hubert(
    hparams, encoder_components, tokenizer, wav_file, device="cpu"
):
    """
    Run inference on a single wav file using HuBERT model.
    """
    hubert = encoder_components["hubert"]

    try:
        # Load and preprocess audio
        signal = load_audio(wav_file, target_sr=hparams["sample_rate"])
        signal = signal.unsqueeze(0).to(device)  # Add batch dimension

        # Calculate actual length in samples
        signal_length = torch.tensor([signal.shape[1]]).to(device)
        # Normalize to relative length (0-1)
        rel_length = signal_length / signal_length.float()

        # Run through HuBERT encoder
        with torch.no_grad():
            hubert_out = hubert(signal, rel_length)

            # Decode using beam search
            hyps, _, _, _ = hparams["beam_searcher"](hubert_out, rel_length)

            # Get the best hypothesis
            predicted_tokens = hyps[0]

            # Decode tokens to text
            predicted_semantics = tokenizer.decode_ids(predicted_tokens)

        return predicted_semantics

    except Exception as e:
        print(f"Error processing {wav_file}: {e}", file=sys.stderr)
        return f"ERROR: {str(e)}"


def read_test_csv(csv_path):
    """
    Read test.csv file and return list of samples.
    """
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(
                {
                    "wav": row["wav"],
                    "transcript": row["transcript"],
                    "semantics": row["semantics"],
                }
            )
    return samples


def write_results_csv(output_path, results):
    """
    Write results to CSV file.
    """
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "wav",
            "transcript",
            "ground_truth_semantics",
            "predicted_semantics",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)


def main():
    parser = argparse.ArgumentParser(
        description="Unified SLU Evaluation Script for CRDNN and HuBERT models"
    )
    parser.add_argument(
        "--test_csv", required=True, type=str, help="Path to test.csv file"
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        type=str,
        help="Path to output results.csv file",
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

    # Check if GPU is available
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        device = "cpu"

    print(f"Using device: {device}")

    # Load hyperparameters
    print(f"Loading hyperparameters from: {args.hparams}")
    with open(args.hparams) as f:
        hparams = load_hyperpyyaml(f)

    # Determine model type and setup accordingly
    model_type = get_model_type(hparams)
    print(f"Detected model type: {model_type}")

    # Setup model based on type
    print("Loading model...")
    if model_type == "crdnn":
        tokenizer, encoder_components = setup_crdnn_model(hparams, device)
        run_inference = run_inference_crdnn
    else:  # hubert
        tokenizer, encoder_components = setup_hubert_model(hparams, device)
        run_inference = run_inference_hubert

    # Read test samples
    print(f"Reading test samples from: {args.test_csv}")
    samples = read_test_csv(args.test_csv)
    print(f"Found {len(samples)} samples")

    # Run inference on all samples
    results = []
    print("\nRunning inference...")

    for sample in tqdm(samples, desc="Processing"):
        wav_file = sample["wav"]

        # Check if file exists
        if not os.path.exists(wav_file):
            print(f"Warning: File not found: {wav_file}")
            predicted_semantics = "ERROR: File not found"
        else:
            predicted_semantics = run_inference(
                hparams, encoder_components, tokenizer, wav_file, device
            )

        results.append(
            {
                "wav": wav_file,
                "transcript": sample["transcript"],
                "ground_truth_semantics": sample["semantics"],
                "predicted_semantics": predicted_semantics,
            }
        )

    # Write results
    print(f"\nWriting results to: {args.output_csv}")
    write_results_csv(args.output_csv, results)

    print("BATCH EVALUATION COMPLETED")
    print(f"Processed {len(results)} samples")
    print(f"Results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
