"""
Script to profile FLOPs/MACs and parameters for SLU models using ptflops. This script profiles the computational 
requirements of the SLU models with a random 1-second audio tensor at 16kHz sampling rate.

Usage:
    python compute_macs.py --hparams <path_to_hparams> --device <cpu|cuda> --compute-macs
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Any, Callable

import torch
import torch.nn as nn
from hyperpyyaml import load_hyperpyyaml
from ptflops import get_model_complexity_info

# Suppress warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# Constants for 1 second of audio at 16kHz
SAMPLE_RATE = 16000
AUDIO_DURATION = 1.0  # seconds
NUM_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)

# Sequence lengths to profile for decoder
DECODER_SEQ_LENGTHS = [16, 32, 64]


def format_macs(macs: int | float) -> str:
    """Format MACs value with proper units."""
    if macs >= 1e9:
        return f"{macs:,.0f} ({macs / 1e9:.2f} GMACs)"
    elif macs >= 1e6:
        return f"{macs:,.0f} ({macs / 1e6:.2f} MMACs)"
    else:
        return f"{macs:,.0f}"


def format_params(params: int | float) -> str:
    """Format parameters value with proper units."""
    if params >= 1e6:
        return f"{params:,.0f} ({params / 1e6:.2f} M)"
    elif params >= 1e3:
        return f"{params:,.0f} ({params / 1e3:.2f} K)"
    else:
        return f"{params:,.0f}"


class HuBERTEncoderWrapper(nn.Module):
    """Wrapper for HuBERT encoder for profiling."""

    def __init__(self, hubert: nn.Module) -> None:
        super().__init__()
        self.hubert = hubert

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        rel_length = torch.ones(batch_size, device=x.device)
        return self.hubert(x, rel_length)


class CRDNNEncoderWrapper(nn.Module):
    """Wrapper for CRDNN ASR encoder for profiling."""

    def __init__(self, asr_encoder: nn.Module, slu_enc: nn.Module) -> None:
        super().__init__()
        self.asr_encoder = asr_encoder
        self.slu_enc = slu_enc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        rel_length = torch.ones(batch_size, device=x.device)
        asr_encoded = self.asr_encoder(x, lengths=rel_length)
        return self.slu_enc(asr_encoded)


class ASREncoderWrapper(nn.Module):
    """Wrapper for just the ASR encoder (CRDNN) for profiling."""

    def __init__(self, asr_encoder: nn.Module) -> None:
        super().__init__()
        self.asr_encoder = asr_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        rel_length = torch.ones(batch_size, device=x.device)
        return self.asr_encoder(x, lengths=rel_length)


class SLUEncoderWrapper(nn.Module):
    """Wrapper for SLU encoder for profiling."""

    def __init__(self, slu_enc: nn.Module, asr_encoder_dim: int) -> None:
        super().__init__()
        self.slu_enc = slu_enc
        self.asr_encoder_dim = asr_encoder_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.slu_enc(x)


class DecoderWrapper(nn.Module):
    """
    Wrapper for decoder components for profiling.

    This wrapper profiles the decoder for a given sequence length,
    simulating the full decoding process using the AttentionalRNNDecoder's
    forward method which processes the entire input sequence at once.
    """

    def __init__(
        self,
        output_emb: nn.Module,
        dec: nn.Module,
        seq_lin: nn.Module,
        encoder_dim: int,
        seq_length: int,
    ) -> None:
        super().__init__()
        self.output_emb = output_emb
        self.dec = dec
        self.seq_lin = seq_lin
        self.encoder_dim = encoder_dim
        self.seq_length = seq_length

    def forward(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """
        Forward pass simulating decoding for seq_length steps.

        Args:
            encoder_out: Encoder output of shape (batch, enc_time, enc_dim)

        Returns:
            Output logits of shape (batch, seq_length, vocab_size)
        """
        batch_size = encoder_out.shape[0]
        device = encoder_out.device
        rel_length = torch.ones(batch_size, device=device)

        # Create input tokens for the full sequence (simulating teacher forcing)
        # Shape: (batch, seq_length)
        tokens = torch.zeros(
            batch_size, self.seq_length, dtype=torch.long, device=device
        )
        # Embed tokens: (batch, seq_length, emb_dim)
        embedded = self.output_emb(tokens)

        # Run decoder for the full sequence
        # AttentionalRNNDecoder.forward() signature: (inp_tensor, enc_states, wav_len)
        dec_out, _ = self.dec(embedded, encoder_out, rel_length)

        # Linear projection to output vocabulary
        # dec_out shape: (batch, seq_length, dec_neurons)
        output = self.seq_lin(dec_out)

        return output


class HuBERTSLUModel(nn.Module):
    """
    Wrapper model for HuBERT-based SLU for profiling.

    This model wraps the HuBERT encoder and the SLU decoder components
    to enable profiling with ptflops.
    """

    def __init__(
        self,
        hubert: nn.Module,
        output_emb: nn.Module,
        dec: nn.Module,
        seq_lin: nn.Module,
    ) -> None:
        super().__init__()
        self.hubert = hubert
        self.output_emb = output_emb
        self.dec = dec
        self.seq_lin = seq_lin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for profiling.

        Args:
            x: Input audio tensor of shape (batch, time)

        Returns:
            Output logits
        """
        batch_size = x.shape[0]
        rel_length = torch.ones(batch_size, device=x.device)

        # HuBERT encoding
        encoded = self.hubert(x, rel_length)

        # For profiling, we simulate one decoder step
        # Create a dummy target token (BOS token = 0)
        # Shape: (batch, 1)
        dummy_token = torch.zeros(batch_size, 1, dtype=torch.long, device=x.device)
        # Embed: (batch, 1, emb_dim)
        embedded = self.output_emb(dummy_token)

        # Run decoder
        # AttentionalRNNDecoder.forward() signature: (inp_tensor, enc_states, wav_len)
        dec_out, _ = self.dec(embedded, encoded, rel_length)

        # Linear projection to output vocabulary
        output = self.seq_lin(dec_out)

        return output


class CRDNNSLUModel(nn.Module):
    """
    Wrapper model for CRDNN-based SLU for profiling.

    This model wraps the ASR encoder, SLU encoder, and decoder components
    to enable profiling with ptflops.
    """

    def __init__(
        self,
        asr_encoder: nn.Module,
        slu_enc: nn.Module,
        output_emb: nn.Module,
        dec: nn.Module,
        seq_lin: nn.Module,
    ) -> None:
        super().__init__()
        self.asr_encoder = asr_encoder
        self.slu_enc = slu_enc
        self.output_emb = output_emb
        self.dec = dec
        self.seq_lin = seq_lin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for profiling.

        Args:
            x: Input audio tensor of shape (batch, time)

        Returns:
            Output logits
        """
        batch_size = x.shape[0]
        rel_length = torch.ones(batch_size, device=x.device)

        # ASR encoder
        asr_encoded = self.asr_encoder(x, lengths=rel_length)

        # SLU encoder
        encoded = self.slu_enc(asr_encoded)

        # For profiling, we simulate one decoder step
        # Create a dummy target token (BOS token = 0)
        # Shape: (batch, 1)
        dummy_token = torch.zeros(batch_size, 1, dtype=torch.long, device=x.device)
        # Embed: (batch, 1, emb_dim)
        embedded = self.output_emb(dummy_token)

        # Run decoder
        # AttentionalRNNDecoder.forward() signature: (inp_tensor, enc_states, wav_len)
        dec_out, _ = self.dec(embedded, encoded, rel_length)

        # Linear projection to output vocabulary
        output = self.seq_lin(dec_out)

        return output


def get_model_type(hparams: dict[str, Any]) -> str:
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


def run_profiling(
    model: nn.Module,
    input_shape: tuple[int, ...],
    input_constructor: Callable[[tuple[int, ...]], torch.Tensor],
    title: str,
) -> tuple[int, int]:
    """
    Run ptflops profiling on a model.

    Returns:
        Tuple of (macs, params) as integers
    """
    print(f"\n{title}")
    print("-" * 70)

    result = get_model_complexity_info(
        model,
        input_shape,
        input_constructor=input_constructor,  # type: ignore[arg-type]
        as_strings=False,
        print_per_layer_stat=True,
        verbose=True,
    )

    # Handle the return value - ptflops returns (macs, params)
    macs = int(result[0]) if result[0] is not None else 0
    params = int(result[1]) if result[1] is not None else 0

    return macs, params


def profile_decoder_seq_lengths(
    output_emb: nn.Module,
    dec: nn.Module,
    seq_lin: nn.Module,
    encoder_dim: int,
    enc_time: int,
    device: str,
) -> None:
    """
    Profile decoder for different sequence lengths.
    """
    print("\n" + "=" * 70)
    print("DECODER PROFILING FOR DIFFERENT SEQUENCE LENGTHS")
    print("=" * 70)

    for seq_len in DECODER_SEQ_LENGTHS:
        decoder_wrapper = DecoderWrapper(output_emb, dec, seq_lin, encoder_dim, seq_len)
        decoder_wrapper.to(device)
        decoder_wrapper.eval()

        # Input shape for decoder: (batch, enc_time, encoder_dim)
        input_shape = (1, enc_time, encoder_dim)

        def input_constructor(
            shape: tuple[int, ...], dev: str = device
        ) -> torch.Tensor:
            return torch.randn(shape, device=dev)

        macs, params = run_profiling(
            decoder_wrapper,
            input_shape,
            input_constructor,
            f"Decoder (seq_length={seq_len})",
        )

        print(f"\n--- Summary for seq_length={seq_len} ---")
        print(f"  MACs: {format_macs(macs)}")
        print(f"  FLOPs (approx 2x MACs): {format_macs(2 * macs)}")
        print(f"  Parameters: {format_params(params)}")
        print(f"  MACs per token: {format_macs(macs / seq_len)}")


def profile_hubert_model(hparams: dict[str, Any], device: str = "cpu") -> None:
    """
    Profile the HuBERT-based SLU model with separate components:
    - HuBERT Encoder
    - SLU Decoder
    """
    print("=" * 70)
    print("PROFILING HuBERT-based SLU Model")
    print("=" * 70)
    print(f"Input: 1 second of audio at {SAMPLE_RATE}Hz ({NUM_SAMPLES} samples)")
    print(f"Device: {device}")
    print("=" * 70)

    # Get model components
    hubert = hparams["hubert"]
    output_emb = hparams["output_emb"]
    dec = hparams["dec"]
    seq_lin = hparams["seq_lin"]
    encoder_dim = hparams["encoder_dim"]

    # Move to device
    hubert.to(device)
    output_emb.to(device)
    dec.to(device)
    seq_lin.to(device)

    def audio_input_constructor(
        input_shape: tuple[int, ...], dev: str = device
    ) -> torch.Tensor:
        return torch.randn(input_shape, device=dev)

    # ========== 1. Profile HuBERT Encoder ==========
    print("\n" + "=" * 70)
    print("1. HuBERT ENCODER")
    print("=" * 70)

    encoder_wrapper = HuBERTEncoderWrapper(hubert)
    encoder_wrapper.to(device)
    encoder_wrapper.eval()

    encoder_macs, encoder_params = run_profiling(
        encoder_wrapper,
        (1, NUM_SAMPLES),
        audio_input_constructor,
        "HuBERT Encoder",
    )

    print("\n--- HuBERT Encoder Summary ---")
    print(f"  MACs: {format_macs(encoder_macs)}")
    print(f"  FLOPs (approx 2x MACs): {format_macs(2 * encoder_macs)}")
    print(f"  Parameters: {format_params(encoder_params)}")

    # Calculate encoder output time dimension
    # HuBERT downsamples by factor of 320 (conv layers)
    enc_time = NUM_SAMPLES // 320

    # ========== 2. Profile SLU Decoder for different sequence lengths ==========
    decoder_results = profile_decoder_seq_lengths_with_totals(
        output_emb, dec, seq_lin, encoder_dim, enc_time, device
    )

    # ========== Total Model Summary ==========
    print("\n" + "=" * 70)
    print("TOTAL MODEL SUMMARY")
    print("=" * 70)

    # Parameter counts
    hubert_encoder_params = sum(p.numel() for p in hubert.parameters())
    decoder_params = (
        sum(p.numel() for p in output_emb.parameters())
        + sum(p.numel() for p in dec.parameters())
        + sum(p.numel() for p in seq_lin.parameters())
    )
    total_params = hubert_encoder_params + decoder_params

    print("\n--- Component Breakdown ---")
    print("\n  HuBERT Encoder:")
    print(f"    MACs: {format_macs(encoder_macs)}")
    print(f"    FLOPs: {format_macs(2 * encoder_macs)}")
    print(f"    Parameters: {format_params(hubert_encoder_params)}")

    print("\n  SLU Decoder (parameters same for all seq lengths):")
    print(f"    Parameters: {format_params(decoder_params)}")
    for seq_len, (dec_macs, _) in decoder_results.items():
        print(
            f"    seq_length={seq_len}: MACs={format_macs(dec_macs)}, FLOPs={format_macs(2 * dec_macs)}"
        )

    print("\n--- Totals (Encoder + Decoder) ---")
    print(f"\n  Total Encoder MACs: {format_macs(encoder_macs)}")
    print(f"  Total Encoder FLOPs: {format_macs(2 * encoder_macs)}")

    # Show total for each decoder sequence length
    for seq_len, (dec_macs, _) in decoder_results.items():
        total_macs = encoder_macs + dec_macs
        print(f"\n  With Decoder seq_length={seq_len}:")
        print(f"    Total MACs: {format_macs(total_macs)}")
        print(f"    Total FLOPs: {format_macs(2 * total_macs)}")

    print(f"\n  Total Parameters: {format_params(total_params)}")
    print("=" * 70)


def profile_decoder_seq_lengths_with_totals(
    output_emb: nn.Module,
    dec: nn.Module,
    seq_lin: nn.Module,
    encoder_dim: int,
    enc_time: int,
    device: str,
) -> dict[int, tuple[int, int]]:
    """
    Profile decoder for different sequence lengths and return results.

    Returns:
        Dictionary mapping seq_length to (macs, params) tuple
    """
    print("\n" + "=" * 70)
    print("SLU DECODER PROFILING FOR DIFFERENT SEQUENCE LENGTHS")
    print("=" * 70)

    results: dict[int, tuple[int, int]] = {}

    for seq_len in DECODER_SEQ_LENGTHS:
        decoder_wrapper = DecoderWrapper(output_emb, dec, seq_lin, encoder_dim, seq_len)
        decoder_wrapper.to(device)
        decoder_wrapper.eval()

        # Input shape for decoder: (batch, enc_time, encoder_dim)
        input_shape = (1, enc_time, encoder_dim)

        def input_constructor(
            shape: tuple[int, ...], dev: str = device
        ) -> torch.Tensor:
            return torch.randn(shape, device=dev)

        macs, params = run_profiling(
            decoder_wrapper,
            input_shape,
            input_constructor,
            f"SLU Decoder (seq_length={seq_len})",
        )

        results[seq_len] = (macs, params)

        print(f"\n--- Summary for seq_length={seq_len} ---")
        print(f"  MACs: {format_macs(macs)}")
        print(f"  FLOPs (approx 2x MACs): {format_macs(2 * macs)}")
        print(f"  Parameters: {format_params(params)}")
        print(f"  MACs per token: {format_macs(macs / seq_len)}")

    return results


def profile_crdnn_model(hparams: dict[str, Any], device: str = "cpu") -> None:
    """
    Profile the CRDNN-based SLU model with separate components:
    - CRDNN ASR Encoder
    - SLU Encoder
    - SLU Decoder
    """
    from speechbrain.inference import EncoderDecoderASR

    print("=" * 70)
    print("PROFILING CRDNN-based SLU Model")
    print("=" * 70)
    print(f"Input: 1 second of audio at {SAMPLE_RATE}Hz ({NUM_SAMPLES} samples)")
    print(f"Device: {device}")
    print("=" * 70)

    # Load the pretrained ASR encoder
    print("Loading ASR encoder from:", hparams["asr_model_path"])
    asr_model = EncoderDecoderASR.from_hparams(
        source=hparams["asr_model_path"],
        savedir=hparams["asr_model_path"],
    )
    asr_encoder = asr_model.mods.encoder  # type: ignore[union-attr]
    asr_encoder.eval()
    asr_encoder.to(device)

    # Get SLU model components
    slu_enc = hparams["slu_enc"]
    output_emb = hparams["output_emb"]
    dec = hparams["dec"]
    seq_lin = hparams["seq_lin"]
    encoder_dim = hparams["encoder_dim"]
    asr_encoder_dim = hparams["ASR_encoder_dim"]

    # Move to device
    slu_enc.to(device)
    output_emb.to(device)
    dec.to(device)
    seq_lin.to(device)

    def audio_input_constructor(
        input_shape: tuple[int, ...], dev: str = device
    ) -> torch.Tensor:
        return torch.randn(input_shape, device=dev)

    # ========== 1. Profile CRDNN ASR Encoder ==========
    print("\n" + "=" * 70)
    print("1. CRDNN ASR ENCODER")
    print("=" * 70)

    asr_wrapper = ASREncoderWrapper(asr_encoder)
    asr_wrapper.to(device)
    asr_wrapper.eval()

    asr_macs, asr_params = run_profiling(
        asr_wrapper,
        (1, NUM_SAMPLES),
        audio_input_constructor,
        "CRDNN ASR Encoder",
    )

    print("\n--- CRDNN ASR Encoder Summary ---")
    print(f"  MACs: {format_macs(asr_macs)}")
    print(f"  FLOPs (approx 2x MACs): {format_macs(2 * asr_macs)}")
    print(f"  Parameters: {format_params(asr_params)}")

    # Calculate ASR encoder output time dimension
    # CRDNN downsamples: 101 frames for 1 second at 16kHz with default settings
    # This is approximately NUM_SAMPLES / 160
    asr_enc_time = NUM_SAMPLES // 160  # ~100 frames

    # ========== 2. Profile SLU Encoder ==========
    print("\n" + "=" * 70)
    print("2. SLU ENCODER")
    print("=" * 70)

    slu_enc_wrapper = SLUEncoderWrapper(slu_enc, asr_encoder_dim)
    slu_enc_wrapper.to(device)
    slu_enc_wrapper.eval()

    # Input shape for SLU encoder: (batch, time, asr_encoder_dim)
    slu_input_shape = (1, asr_enc_time, asr_encoder_dim)

    def slu_input_constructor(
        shape: tuple[int, ...], dev: str = device
    ) -> torch.Tensor:
        return torch.randn(shape, device=dev)

    slu_enc_macs, slu_enc_params = run_profiling(
        slu_enc_wrapper,
        slu_input_shape,
        slu_input_constructor,
        "SLU Encoder",
    )

    print("\n--- SLU Encoder Summary ---")
    print(f"  MACs: {format_macs(slu_enc_macs)}")
    print(f"  FLOPs (approx 2x MACs): {format_macs(2 * slu_enc_macs)}")
    print(f"  Parameters: {format_params(slu_enc_params)}")

    # SLU encoder output time dimension (same as input for LSTM)
    enc_time = asr_enc_time

    # ========== 3. Profile SLU Decoder for different sequence lengths ==========
    decoder_results = profile_decoder_seq_lengths_with_totals(
        output_emb, dec, seq_lin, encoder_dim, enc_time, device
    )

    # ========== Total Model Summary ==========
    print("\n" + "=" * 70)
    print("TOTAL MODEL SUMMARY")
    print("=" * 70)

    # Parameter counts
    asr_encoder_params = sum(p.numel() for p in asr_encoder.parameters())
    slu_encoder_params = sum(p.numel() for p in slu_enc.parameters())
    decoder_params = (
        sum(p.numel() for p in output_emb.parameters())
        + sum(p.numel() for p in dec.parameters())
        + sum(p.numel() for p in seq_lin.parameters())
    )
    total_params = asr_encoder_params + slu_encoder_params + decoder_params

    # Encoder total (ASR + SLU)
    total_encoder_macs = asr_macs + slu_enc_macs

    print("\n--- Component Breakdown ---")
    print("\n  CRDNN ASR Encoder:")
    print(f"    MACs: {format_macs(asr_macs)}")
    print(f"    FLOPs: {format_macs(2 * asr_macs)}")
    print(f"    Parameters: {format_params(asr_encoder_params)}")

    print("\n  SLU Encoder:")
    print(f"    MACs: {format_macs(slu_enc_macs)}")
    print(f"    FLOPs: {format_macs(2 * slu_enc_macs)}")
    print(f"    Parameters: {format_params(slu_encoder_params)}")

    print("\n  SLU Decoder (parameters same for all seq lengths):")
    print(f"    Parameters: {format_params(decoder_params)}")
    for seq_len, (dec_macs, _) in decoder_results.items():
        print(
            f"    seq_length={seq_len}: MACs={format_macs(dec_macs)}, FLOPs={format_macs(2 * dec_macs)}"
        )

    print("\n--- Totals (Encoder + Decoder) ---")
    print(f"\n  Total Encoder MACs (ASR + SLU): {format_macs(total_encoder_macs)}")
    print(f"  Total Encoder FLOPs (ASR + SLU): {format_macs(2 * total_encoder_macs)}")

    # Show total for each decoder sequence length
    for seq_len, (dec_macs, _) in decoder_results.items():
        total_macs = total_encoder_macs + dec_macs
        print(f"\n  With Decoder seq_length={seq_len}:")
        print(f"    Total MACs: {format_macs(total_macs)}")
        print(f"    Total FLOPs: {format_macs(2 * total_macs)}")

    print(f"\n  Total Parameters: {format_params(total_params)}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile FLOPs/MACs for SLU models using ptflops"
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
        help="Device to run profiling on",
    )
    parser.add_argument(
        "--compute-macs",
        action="store_true",
        help="Compute MACs/FLOPs for the model",
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

    # Determine model type
    model_type = get_model_type(hparams)
    print(f"Detected model type: {model_type}")

    if args.compute_macs:
        if model_type == "hubert":
            profile_hubert_model(hparams, device)
        else:  # crdnn
            profile_crdnn_model(hparams, device)
    else:
        print("Use --compute-macs to profile the model")


if __name__ == "__main__":
    main()
