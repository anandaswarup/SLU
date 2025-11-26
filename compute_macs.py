"""
Script to compute MACs (Multiply-Accumulate Operations) and latency for SLU model inference.

This script profiles the computational complexity and inference latency of the encoder
and decoder components of the Spoken Language Understanding model.

Requirements:
    pip install thop

Usage:
    python compute_macs.py --hparams hparams/slu.yaml --device cpu
"""

import argparse
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from hyperpyyaml import load_hyperpyyaml
from thop import profile

# Set environment variable and suppress warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# Import SpeechBrain after setting warnings
from speechbrain.inference.ASR import EncoderDecoderASR  # noqa: E402

# -----------------------------
# WRAPPER CLASSES
# -----------------------------


class ASREncoderWrapper(nn.Module):
    """Wrapper for ASR encoder to make it compatible with THOP profiling."""

    def __init__(self, asr_model):
        super().__init__()
        self.asr_model = asr_model

    def forward(self, wavs, wav_lens):
        """Forward pass through ASR encoder."""
        return self.asr_model.encode_batch(wavs, wav_lens)


class SLUEncoderWrapper(nn.Module):
    """Wrapper for SLU encoder (LSTM + Linear) to make it compatible with THOP profiling."""

    def __init__(self, slu_enc):
        super().__init__()
        self.slu_enc = slu_enc

    def forward(self, encoder_out):
        """Forward pass through SLU encoder."""
        return self.slu_enc(encoder_out)


class SLUDecoderWrapper(nn.Module):
    """Wrapper for SLU decoder to make it compatible with THOP profiling."""

    def __init__(self, output_emb, dec, seq_lin, encoder_out, wav_lens, seq_length):
        super().__init__()
        self.output_emb = output_emb
        self.dec = dec
        self.seq_lin = seq_lin
        self.encoder_out = encoder_out
        self.wav_lens = wav_lens
        self.seq_length = seq_length

    def forward(self, tokens_bos):
        """
        Forward pass through decoder.

        Args:
            tokens_bos: Beginning of sequence tokens [batch_size, seq_length]
        """
        # Embedding layer
        e_in = self.output_emb(tokens_bos)

        # Decoder (GRU with attention)
        h, _ = self.dec(e_in, self.encoder_out, self.wav_lens)

        # Output linear layer
        logits = self.seq_lin(h)

        return logits


def compute_macs(hparams_file, device="cpu"):
    """
    Compute MACs for the SLU model.

    Args:
        hparams_file: Path to hyperparameters YAML file
        device: Device to run on ("cpu" or "cuda")
    """
    # Load hyperparameters
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    # Load tokenizer
    tokenizer = hparams["tokenizer"]
    tokenizer.load(hparams["tokenizer_file"])

    # Load pretrained ASR model
    print(f"Loading ASR model from: {hparams['asr_model_path']}")
    asr_model = EncoderDecoderASR.from_hparams(
        source=hparams["asr_model_path"],
        savedir=hparams["asr_model_path"],
        run_opts={"device": device},
    )
    asr_model.mods.eval()  # type: ignore

    # Load SLU model components
    slu_enc = hparams["slu_enc"].to(device)
    output_emb = hparams["output_emb"].to(device)
    dec = hparams["dec"].to(device)
    seq_lin = hparams["seq_lin"].to(device)

    # Set all models to eval mode
    slu_enc.eval()
    output_emb.eval()
    dec.eval()
    seq_lin.eval()

    # -----------------------------
    # Prepare dummy input: 1 sec audio (16kHz)
    # -----------------------------
    batch_size = 1
    audio_length = 16000
    sample_rate = hparams["sample_rate"]

    print(f"\n{'=' * 60}")
    print(
        f"Computing MACs for {audio_length / sample_rate:.1f} second audio @ {sample_rate}Hz"
    )
    print(f"{'=' * 60}\n")

    audio = torch.randn(batch_size, audio_length).to(device)
    wav_lens = torch.tensor([1.0]).to(device)  # Relative length

    # -----------------------------
    # Step 1: ASR Encoder
    # -----------------------------
    print("Step 1: Profiling ASR Encoder...")
    with torch.no_grad():
        asr_encoder_out = asr_model.encode_batch(audio, wav_lens)  # type: ignore

    print(f"ASR Encoder output shape: {asr_encoder_out.shape}")

    asr_encoder_wrapper = ASREncoderWrapper(asr_model)
    macs_asr_enc, params_asr_enc = profile(  # type: ignore
        asr_encoder_wrapper, inputs=(audio, wav_lens), verbose=False
    )

    print(f"  → MACs: {macs_asr_enc / 1e6:.2f} M")
    print(f"  → Params: {params_asr_enc / 1e6:.2f} M")

    # -----------------------------
    # Step 2: SLU Encoder (LSTM + Linear)
    # -----------------------------
    print("\nStep 2: Profiling SLU Encoder (LSTM + Linear)...")
    with torch.no_grad():
        slu_encoder_out = slu_enc(asr_encoder_out)

    print(f"SLU Encoder output shape: {slu_encoder_out.shape}")

    slu_encoder_wrapper = SLUEncoderWrapper(slu_enc)
    macs_slu_enc, params_slu_enc = profile(  # type: ignore
        slu_encoder_wrapper, inputs=(asr_encoder_out,), verbose=False
    )

    print(f"  → MACs: {macs_slu_enc / 1e6:.2f} M")
    print(f"  → Params: {params_slu_enc / 1e6:.2f} M")

    # -----------------------------
    # Step 3: SLU Decoder (varying sequence lengths)
    # -----------------------------
    print("\nStep 3: Profiling SLU Decoder (GRU + Attention + Linear)...")
    print(f"{'=' * 60}")

    # Test with different output sequence lengths
    seq_lengths = [8, 16, 32, 64]
    params_dec = 0  # Initialize to avoid unbound variable error

    for seq_len in seq_lengths:
        print(f"\nDecoder sequence length: {seq_len}")

        # Create dummy tokens for decoder input
        tokens_bos = torch.zeros(batch_size, seq_len, dtype=torch.long).to(device)

        decoder_wrapper = SLUDecoderWrapper(
            output_emb, dec, seq_lin, slu_encoder_out, wav_lens, seq_len
        )

        macs_dec, params_dec = profile(  # type: ignore
            decoder_wrapper, inputs=(tokens_bos,), verbose=False
        )

        print(f"  → MACs: {macs_dec / 1e6:.2f} M")
        print(f"  → Params: {params_dec / 1e6:.2f} M")

        # Compute total MACs
        total_macs = macs_asr_enc + macs_slu_enc + macs_dec
        print(f"  → Total MACs (ASR + SLU Enc + Dec): {total_macs / 1e6:.2f} M")

    # -----------------------------
    # Summary
    # -----------------------------
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"ASR Encoder MACs:     {macs_asr_enc / 1e6:>10.2f} M")
    print(f"ASR Encoder Params:   {params_asr_enc / 1e6:>10.2f} M")
    print(f"\nSLU Encoder MACs:     {macs_slu_enc / 1e6:>10.2f} M")
    print(f"SLU Encoder Params:   {params_slu_enc / 1e6:>10.2f} M")
    print(f"\nSLU Decoder Params:   {params_dec / 1e6:>10.2f} M")
    print("(Decoder MACs vary with sequence length)")
    print(f"\nTotal Encoder MACs:   {(macs_asr_enc + macs_slu_enc) / 1e6:>10.2f} M")
    print(
        f"Total Params:         {(params_asr_enc + params_slu_enc + params_dec) / 1e6:>10.2f} M"
    )
    print(f"{'=' * 60}\n")


def compute_latency(hparams_file, device="cpu", num_samples=35):
    """
    Compute average latency for the SLU model with varying audio durations.

    Args:
        hparams_file: Path to hyperparameters YAML file
        device: Device to run on ("cpu" or "cuda")
        num_samples: Number of samples to average over (default: 35)
    """
    # Load hyperparameters
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    # Load tokenizer
    tokenizer = hparams["tokenizer"]
    tokenizer.load(hparams["tokenizer_file"])

    # Load pretrained ASR model
    print(f"\n{'=' * 60}")
    print("LATENCY COMPUTATION")
    print(f"{'=' * 60}")
    print(f"Loading ASR model from: {hparams['asr_model_path']}")
    asr_model = EncoderDecoderASR.from_hparams(
        source=hparams["asr_model_path"],
        savedir=hparams["asr_model_path"],
        run_opts={"device": device},
    )
    asr_model.mods.eval()  # type: ignore

    # Load SLU model components
    slu_enc = hparams["slu_enc"].to(device)
    output_emb = hparams["output_emb"].to(device)
    dec = hparams["dec"].to(device)
    seq_lin = hparams["seq_lin"].to(device)
    beam_searcher = hparams["beam_searcher"]

    # Set all models to eval mode
    slu_enc.eval()
    output_emb.eval()
    dec.eval()
    seq_lin.eval()

    sample_rate = hparams["sample_rate"]
    batch_size = 1

    # Audio duration range: 1-3 seconds
    min_duration = 1.0
    max_duration = 3.0

    print(f"\nRunning {num_samples} inference samples")
    print(f"Audio duration range: {min_duration}-{max_duration} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Device: {device}\n")

    # Warm-up run
    print("Performing warm-up run...")
    audio_length = int(1.0 * sample_rate)
    audio = torch.randn(batch_size, audio_length).to(device)
    wav_lens = torch.tensor([1.0]).to(device)

    with torch.no_grad():
        asr_encoder_out = asr_model.encode_batch(audio, wav_lens)  # type: ignore
        slu_encoder_out = slu_enc(asr_encoder_out)
        hyps, _, _, _ = beam_searcher(slu_encoder_out, wav_lens)

    print("Warm-up complete.\n")

    # Collect latencies for different audio durations
    latencies = []
    durations = []

    print(f"{'Sample':<8} {'Duration (s)':<15} {'Latency (ms)':<15} {'RTF':<10}")
    print("-" * 60)

    for i in range(num_samples):
        # Random audio duration between 1-3 seconds
        duration = np.random.uniform(min_duration, max_duration)
        audio_length = int(duration * sample_rate)

        # Generate random audio
        audio = torch.randn(batch_size, audio_length).to(device)
        wav_lens = torch.tensor([1.0]).to(device)

        # Measure inference time
        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        with torch.no_grad():
            # ASR Encoder
            asr_encoder_out = asr_model.encode_batch(audio, wav_lens)  # type: ignore

            # SLU Encoder
            slu_encoder_out = slu_enc(asr_encoder_out)

            # Beam search decoder
            hyps, _, _, _ = beam_searcher(slu_encoder_out, wav_lens)

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds

        # Real-Time Factor (RTF) = processing_time / audio_duration
        rtf = (end_time - start_time) / duration

        latencies.append(latency)
        durations.append(duration)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"{i + 1:<8} {duration:<15.2f} {latency:<15.2f} {rtf:<10.4f}")

    # Compute statistics
    latencies = np.array(latencies)
    durations = np.array(durations)

    # Average latency normalized to 1 second audio
    # latency_per_sec[i] = latencies[i] / durations[i]
    latency_per_sec = latencies / durations
    avg_latency_1sec = np.mean(latency_per_sec)
    std_latency_1sec = np.std(latency_per_sec)
    min_latency_1sec = np.min(latency_per_sec)
    max_latency_1sec = np.max(latency_per_sec)

    # Average RTF
    rtf_values = latencies / 1000.0 / durations
    avg_rtf = np.mean(rtf_values)

    print("\n" + "=" * 60)
    print("LATENCY STATISTICS (normalized to 1 second audio)")
    print("=" * 60)
    print(f"Number of samples:    {num_samples}")
    print(f"Audio duration range: {min_duration:.1f}-{max_duration:.1f} seconds")
    print(f"\nAverage latency:      {avg_latency_1sec:.2f} ± {std_latency_1sec:.2f} ms")
    print(f"Min latency:          {min_latency_1sec:.2f} ms")
    print(f"Max latency:          {max_latency_1sec:.2f} ms")
    print(f"Average RTF:          {avg_rtf:.4f}")
    print("=" * 60 + "\n")

    return avg_latency_1sec, std_latency_1sec, avg_rtf


def main():
    """Main function to parse arguments and run MAC and latency computation."""
    parser = argparse.ArgumentParser(
        description="Compute MACs and latency for SLU model inference"
    )
    parser.add_argument(
        "--hparams",
        type=str,
        required=True,
        help="Path to hyperparameters YAML file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on (cpu or cuda)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=35,
        help="Number of samples for latency computation (default: 35)",
    )
    parser.add_argument(
        "--skip-macs",
        action="store_true",
        help="Skip MACs computation and only compute latency",
    )
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Skip latency computation and only compute MACs",
    )

    args = parser.parse_args()

    # Check if CUDA is available when requested
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Run MAC computation
    if not args.skip_macs:
        compute_macs(args.hparams, args.device)

    # Run latency computation
    if not args.skip_latency:
        compute_latency(args.hparams, args.device, args.num_samples)


if __name__ == "__main__":
    main()
