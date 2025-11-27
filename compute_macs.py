"""
Script to compute MACs (Multiply-Accumulate Operations) and latency for SLU model inference.

This script profiles the computational complexity and inference latency of the encoder
and decoder components of the Spoken Language Understanding model.

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

# Set environment variable and suppress warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# Import SpeechBrain after setting warnings
from speechbrain.inference.ASR import EncoderDecoderASR  # noqa: E402

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------


def profile_model(model, *sample_inputs, device="cpu"):
    """
    Profile a model using ptflops to get MACs and parameters.
    Compatible with thop-style API.

    Args:
        model: PyTorch model to profile
        *sample_inputs: Sample input tensors
        device: Device to run on

    Returns:
        tuple: (macs, params) in integer values
    """
    model = model.to(device)
    model.eval()

    try:
        from ptflops.flops_counter import get_model_complexity_info as gmci

        # Count parameters manually
        params = sum(p.numel() for p in model.parameters())

        # Handle case with no inputs
        if len(sample_inputs) == 0:
            return 0, params

        # For models with multiple inputs, create a wrapper that takes single input
        if len(sample_inputs) > 1:
            # Create a wrapper class that stores the extra inputs
            class SingleInputWrapper(nn.Module):
                def __init__(self, wrapped_model, extra_inputs):
                    super().__init__()
                    self.wrapped_model = wrapped_model
                    self.extra_inputs = extra_inputs

                def forward(self, x):
                    return self.wrapped_model(x, *self.extra_inputs)

            wrapped_model = SingleInputWrapper(model, sample_inputs[1:])
            first_input = sample_inputs[0]
        else:
            wrapped_model = model
            first_input = sample_inputs[0]

        # Get input shape
        input_shape = (
            tuple(first_input.shape[1:])
            if len(first_input.shape) > 1
            else (first_input.shape[0],)
        )

        with torch.no_grad():
            try:
                macs_result, params_result = gmci(
                    wrapped_model,
                    input_shape,
                    as_strings=False,
                    print_per_layer_stat=False,
                    verbose=False,
                )

                # Ensure we return integers
                if isinstance(macs_result, (int, float)):
                    macs = int(macs_result)
                else:
                    macs = 0

                if isinstance(params_result, (int, float)):
                    params = int(params_result)

                return macs, params
            except Exception:
                # If ptflops fails, return 0 MACs and counted params
                return 0, params

    except Exception:
        # Fallback: count parameters manually and return 0 for MACs
        params = sum(p.numel() for p in model.parameters())
        return 0, params


# -----------------------------
# WRAPPER CLASSES
# -----------------------------


class ASREncoderWrapper(nn.Module):
    """Wrapper for ASR CRDNN encoder to make it compatible with ptflops profiling.

    Note: This only wraps the encoder module, not the entire ASR model.
    The encoder extracts acoustic embeddings which are then passed to the SLU model.
    """

    def __init__(self, asr_encoder):
        super().__init__()
        self.encoder = asr_encoder

    def forward(self, wavs, wav_lens):
        """Forward pass through ASR CRDNN encoder only."""
        return self.encoder(wavs, wav_lens)


class SLUEncoderWrapper(nn.Module):
    """Wrapper for SLU encoder (LSTM + Linear) to make it compatible with ptflops profiling."""

    def __init__(self, slu_enc):
        super().__init__()
        self.slu_enc = slu_enc

    def forward(self, encoder_out):
        """Forward pass through SLU encoder."""
        return self.slu_enc(encoder_out)


class LSTMWrapper(nn.Module):
    """Wrapper for individual LSTM layer."""

    def __init__(self, lstm):
        super().__init__()
        self.lstm = lstm

    def forward(self, x):
        """Forward pass through LSTM."""
        return self.lstm(x)


class LinearWrapper(nn.Module):
    """Wrapper for linear layer."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        """Forward pass through linear layer."""
        return self.linear(x)


class EmbeddingWrapper(nn.Module):
    """Wrapper for embedding layer."""

    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, tokens):
        """Forward pass through embedding."""
        return self.embedding(tokens)


class AttentionalRNNDecoderWrapper(nn.Module):
    """Wrapper for AttentionalRNNDecoder (GRU with attention)."""

    def __init__(self, dec, encoder_out, wav_lens):
        super().__init__()
        self.dec = dec
        self.encoder_out = encoder_out
        self.wav_lens = wav_lens

    def forward(self, e_in):
        """Forward pass through decoder."""
        h, _ = self.dec(e_in, self.encoder_out, self.wav_lens)
        return h


class ModuleWrapper(nn.Module):
    """Generic wrapper for any module."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        """Forward pass through module."""
        return self.module(*args, **kwargs)


class SLUDecoderWrapper(nn.Module):
    """Wrapper for SLU decoder to make it compatible with ptflops profiling."""

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
    # Step 1: ASR CRDNN Encoder - Layer-by-Layer Profiling
    # -----------------------------
    print("Step 1: Profiling ASR CRDNN Encoder (acoustic embeddings only)...")
    print(f"{'=' * 60}")

    # Extract only the encoder module from the ASR model
    # The encode_batch method uses mods.encoder which is the CRDNN encoder
    asr_encoder = asr_model.mods.encoder  # type: ignore

    # Get the full output first
    with torch.no_grad():
        asr_encoder_out = asr_model.encode_batch(audio, wav_lens)  # type: ignore

    print(f"\nASR Encoder output shape: {asr_encoder_out.shape}")

    # Profile each layer in the CRDNN encoder
    print("\n1a. Individual Layers in ASR CRDNN Encoder:")
    print("-" * 60)

    # Inspect the encoder structure
    # CRDNN typically has: CNN layers -> RNN layers -> DNN layers
    layer_macs = []
    layer_params = []
    layer_names = []

    # Get intermediate outputs by running through each layer
    current_input = audio
    current_lens = wav_lens

    for idx, (name, layer) in enumerate(asr_encoder.named_children()):
        print(f"\n  Layer {idx}: {name} ({type(layer).__name__})")

        try:
            # Try to get the output first
            with torch.no_grad():
                layer_input = current_input
                layer_lens = current_lens

                try:
                    # Try with lengths parameter
                    layer_out = layer(layer_input, layer_lens)
                    has_lens = True
                except Exception:
                    # Try without lengths parameter
                    try:
                        layer_out = layer(layer_input)
                        has_lens = False
                    except Exception as e:
                        print(f"    → Could not run layer (error: {str(e)[:80]})")
                        continue

                # Update current state
                if isinstance(layer_out, tuple):
                    current_input = layer_out[0]
                    current_lens = layer_out[1] if len(layer_out) > 1 else current_lens
                else:
                    current_input = layer_out

                print(
                    f"    Output shape: {current_input.shape if isinstance(current_input, torch.Tensor) else 'N/A'}"
                )

                # Now profile the layer
                layer_wrapper = ModuleWrapper(layer)
                try:
                    if has_lens:
                        macs, params = profile_model(
                            layer_wrapper, layer_input, layer_lens, device=device
                        )
                    else:
                        macs, params = profile_model(
                            layer_wrapper, layer_input, device=device
                        )

                    # If params are 0, manually count them
                    if params == 0:
                        params = sum(p.numel() for p in layer.parameters())

                    layer_macs.append(macs)
                    layer_params.append(params)
                    layer_names.append(name)

                    print(f"    → MACs: {macs / 1e6:.2f} M")
                    print(f"    → Params: {params / 1e6:.2f} M")
                except Exception:
                    # Profiling failed, just count params
                    params = sum(p.numel() for p in layer.parameters())
                    macs = 0
                    layer_macs.append(macs)
                    layer_params.append(params)
                    layer_names.append(name)
                    print(f"    → MACs: {macs / 1e6:.2f} M (profiling failed)")
                    print(f"    → Params: {params / 1e6:.2f} M")

                # If this is the CRDNN model layer, profile its sub-layers (CNN, RNN, DNN)
                if name == "model" and hasattr(layer, "named_children"):
                    print("\n    Profiling CRDNN sub-layers:")
                    print("    " + "-" * 56)

                    crdnn_input = layer_input
                    prev_output = None  # Track the actual output from previous layer

                    for sub_idx, (sub_name, sub_layer) in enumerate(
                        layer.named_children()
                    ):
                        # Prepare layer description with details
                        layer_details = ""

                        # Add details for different layer types
                        if sub_name == "CNN" and hasattr(sub_layer, "__len__"):
                            # Count number of CNN blocks and detect Conv type
                            num_blocks = len(list(sub_layer.children()))
                            cnn_type = "Conv2d"  # Default assumption for CRDNN
                            # Check first conv layer to determine type
                            for child in sub_layer.modules():
                                if isinstance(child, torch.nn.Conv1d):
                                    cnn_type = "Conv1d"
                                    break
                                elif isinstance(child, torch.nn.Conv2d):
                                    cnn_type = "Conv2d"
                                    break
                            layer_details = f" ({cnn_type}, {num_blocks} blocks)"
                        elif sub_name == "RNN" and hasattr(sub_layer, "rnn"):
                            # Get RNN details from SpeechBrain LSTM wrapper
                            rnn = sub_layer.rnn
                            rnn_type = type(rnn).__name__  # LSTM or GRU
                            num_layers = rnn.num_layers
                            is_bidirectional = rnn.bidirectional
                            hidden_size = rnn.hidden_size
                            direction_str = (
                                "Bidirectional"
                                if is_bidirectional
                                else "Unidirectional"
                            )
                            layer_details = f" ({rnn_type}, {direction_str}, {num_layers} layers, {hidden_size} neurons)"
                        elif sub_name == "DNN" and hasattr(sub_layer, "__len__"):
                            # Count number of DNN blocks
                            layer_details = (
                                f" ({len(list(sub_layer.children()))} blocks)"
                            )

                        print(f"\n      Sub-layer {sub_idx}: {sub_name}{layer_details}")

                        try:
                            with torch.no_grad():
                                # Determine input for this sub-layer
                                if sub_idx == 0:
                                    # First layer (CNN) uses original input
                                    sub_input = crdnn_input
                                elif prev_output is not None:
                                    # Use the actual output from previous layer
                                    sub_input = prev_output
                                else:
                                    # Fallback
                                    sub_input = crdnn_input

                                # Run through the sub-layer to get output shape
                                try:
                                    sub_out = sub_layer(sub_input)
                                except Exception as e:
                                    print(
                                        f"        → Could not run (error: {str(e)[:60]})"
                                    )
                                    continue

                                if isinstance(sub_out, tuple):
                                    prev_output = sub_out[0]
                                else:
                                    prev_output = sub_out

                                print(
                                    f"        Output shape: {prev_output.shape if isinstance(prev_output, torch.Tensor) else 'N/A'}"
                                )

                                # Profile the sub-layer with the correct input
                                sub_wrapper = ModuleWrapper(sub_layer)
                                try:
                                    sub_macs, sub_params = profile_model(
                                        sub_wrapper,
                                        sub_input,
                                        device=device,
                                    )
                                    # If params are 0, manually count them
                                    if sub_params == 0:
                                        sub_params = sum(
                                            p.numel() for p in sub_layer.parameters()
                                        )
                                    print(f"        → MACs: {sub_macs / 1e6:.2f} M")
                                    print(f"        → Params: {sub_params / 1e6:.2f} M")
                                except Exception:
                                    sub_params = sum(
                                        p.numel() for p in sub_layer.parameters()
                                    )
                                    print("        → MACs: 0.00 M (profiling failed)")
                                    print(f"        → Params: {sub_params / 1e6:.2f} M")

                        except Exception as e:
                            print(
                                f"        → Error processing sub-layer: {str(e)[:60]}"
                            )

        except Exception as e:
            print(f"    → Could not process layer (error: {str(e)[:80]})")

    # Profile complete ASR encoder
    print("\n1b. Complete ASR CRDNN Encoder:")
    print("-" * 60)

    # Wrap only the encoder module for profiling
    asr_encoder_wrapper = ASREncoderWrapper(asr_encoder)
    macs_asr_enc, params_asr_enc = profile_model(
        asr_encoder_wrapper, audio, wav_lens, device=device
    )

    # If params_asr_enc is 0, manually count from the encoder
    if params_asr_enc == 0:
        params_asr_enc = sum(p.numel() for p in asr_encoder.parameters())

    print(f"  → Total MACs: {macs_asr_enc / 1e6:.2f} M")
    print(f"  → Total Params: {params_asr_enc / 1e6:.2f} M")

    # Verification if we successfully profiled individual layers
    if layer_macs:
        total_layer_macs = sum(layer_macs)
        total_layer_params = sum(layer_params)
        print(f"  → Verification: Sum of layer MACs = {total_layer_macs / 1e6:.2f} M")
        print(
            f"  → Verification: Sum of layer Params = {total_layer_params / 1e6:.2f} M"
        )

    print("  (Note: Only CRDNN encoder, excluding ASR decoder)")

    # -----------------------------
    # Step 2: SLU Encoder (LSTM + Linear) - Layer-by-Layer Profiling
    # -----------------------------
    print("\nStep 2: Profiling SLU Encoder (LSTM + Linear) - Layer by Layer...")
    print(f"{'=' * 60}")

    # The SLU encoder is a Sequential container with LSTM and Linear
    # Access individual layers by name (SpeechBrain uses named modules)
    layers = list(slu_enc.children())
    lstm_layer = layers[0]  # LSTM layer
    linear_layer = layers[1]  # Linear layer

    # Profile LSTM layer
    # Get LSTM details
    lstm_rnn = lstm_layer.rnn if hasattr(lstm_layer, "rnn") else lstm_layer
    lstm_type = type(lstm_rnn).__name__
    lstm_num_layers = lstm_rnn.num_layers
    is_bidirectional = lstm_rnn.bidirectional
    direction_str = "Bidirectional" if is_bidirectional else "Unidirectional"

    print(f"\n2a. {lstm_type} Layer ({direction_str}, {lstm_num_layers} layers):")
    lstm_wrapper = LSTMWrapper(lstm_layer)

    with torch.no_grad():
        lstm_out, _ = lstm_layer(asr_encoder_out)

    print(f"  LSTM output shape: {lstm_out.shape}")

    macs_lstm, params_lstm = profile_model(lstm_wrapper, asr_encoder_out, device=device)

    print(f"  → MACs: {macs_lstm / 1e6:.2f} M")
    print(f"  → Params: {params_lstm / 1e6:.2f} M")

    # Profile Linear layer
    print("\n2b. Linear Layer (projection):")
    linear_wrapper = LinearWrapper(linear_layer)

    with torch.no_grad():
        linear_out = linear_layer(lstm_out)

    print(f"  Linear output shape: {linear_out.shape}")

    macs_linear, params_linear = profile_model(linear_wrapper, lstm_out, device=device)

    print(f"  → MACs: {macs_linear / 1e6:.2f} M")
    print(f"  → Params: {params_linear / 1e6:.2f} M")

    # Profile complete SLU encoder
    print("\n2c. Complete SLU Encoder (LSTM + Linear):")
    with torch.no_grad():
        slu_encoder_out = slu_enc(asr_encoder_out)

    print(f"  SLU Encoder output shape: {slu_encoder_out.shape}")

    slu_encoder_wrapper = SLUEncoderWrapper(slu_enc)
    macs_slu_enc, params_slu_enc = profile_model(
        slu_encoder_wrapper, asr_encoder_out, device=device
    )

    print(f"  → Total MACs: {macs_slu_enc / 1e6:.2f} M")
    print(f"  → Total Params: {params_slu_enc / 1e6:.2f} M")
    print(
        f"  → Verification: LSTM + Linear MACs = {(macs_lstm + macs_linear) / 1e6:.2f} M"
    )

    # -----------------------------
    # Step 3: SLU Decoder (varying sequence lengths) - Layer-by-Layer Profiling
    # -----------------------------
    print("\n\nStep 3: Profiling SLU Decoder (Embedding + GRU + Attention + Linear)...")
    print(f"{'=' * 60}")

    # Get RNN details from decoder (outside loop for summary section)
    dec_rnn_type = "GRU"  # Default for AttentionalRNNDecoder
    dec_num_layers = 1
    if hasattr(dec, "rnn"):
        dec_rnn_type = type(dec.rnn).__name__ if hasattr(dec.rnn, "__name__") else "GRU"
        dec_num_layers = dec.rnn.num_layers if hasattr(dec.rnn, "num_layers") else 1

    # Test with different output sequence lengths
    seq_lengths = [8, 16, 32, 64]

    # Initialize variables to avoid unbound errors
    params_dec = 0
    params_emb = 0
    params_attn_dec = 0
    params_out_lin = 0
    macs_dec = 0
    macs_emb = 0
    macs_attn_dec = 0
    macs_out_lin = 0

    for seq_len in seq_lengths:
        print(f"\nDecoder sequence length: {seq_len}")
        print("-" * 60)

        # Create dummy tokens for decoder input
        tokens_bos = torch.zeros(batch_size, seq_len, dtype=torch.long).to(device)

        # 3a. Profile Embedding layer
        print("\n3a. Embedding Layer:")
        emb_wrapper = EmbeddingWrapper(output_emb)

        with torch.no_grad():
            e_in = output_emb(tokens_bos)

        print(f"  Embedding output shape: {e_in.shape}")

        macs_emb, params_emb = profile_model(emb_wrapper, tokens_bos, device=device)

        print(f"  → MACs: {macs_emb / 1e6:.2f} M")
        print(f"  → Params: {params_emb / 1e6:.2f} M")

        # 3b. Profile AttentionalRNNDecoder (GRU + Attention)
        print(
            f"\n3b. AttentionalRNNDecoder ({dec_rnn_type} with Attention, {dec_num_layers} layers):"
        )
        attn_dec_wrapper = AttentionalRNNDecoderWrapper(dec, slu_encoder_out, wav_lens)

        with torch.no_grad():
            h, _ = dec(e_in, slu_encoder_out, wav_lens)

        print(f"  Decoder output shape: {h.shape}")

        macs_attn_dec, params_attn_dec = profile_model(
            attn_dec_wrapper, e_in, device=device
        )

        print(f"  → MACs: {macs_attn_dec / 1e6:.2f} M")
        print(f"  → Params: {params_attn_dec / 1e6:.2f} M")

        # 3c. Profile Output Linear layer
        print("\n3c. Output Linear Layer:")
        out_lin_wrapper = LinearWrapper(seq_lin)

        with torch.no_grad():
            logits = seq_lin(h)

        print(f"  Output logits shape: {logits.shape}")

        macs_out_lin, params_out_lin = profile_model(out_lin_wrapper, h, device=device)

        print(f"  → MACs: {macs_out_lin / 1e6:.2f} M")
        print(f"  → Params: {params_out_lin / 1e6:.2f} M")

        # 3d. Profile Complete Decoder
        print("\n3d. Complete Decoder (Emb + Attn + Linear):")

        decoder_wrapper = SLUDecoderWrapper(
            output_emb, dec, seq_lin, slu_encoder_out, wav_lens, seq_len
        )

        macs_dec, params_dec = profile_model(decoder_wrapper, tokens_bos, device=device)

        print(f"  → Total MACs: {macs_dec / 1e6:.2f} M")
        print(f"  → Total Params: {params_dec / 1e6:.2f} M")
        print(
            f"  → Verification: Emb + Decoder + Linear MACs = {(macs_emb + macs_attn_dec + macs_out_lin) / 1e6:.2f} M"
        )

        # Compute total MACs
        total_macs = macs_asr_enc + macs_slu_enc + macs_dec
        print(
            f"\n  → Total System MACs (ASR + SLU Enc + Dec): {total_macs / 1e6:.2f} M"
        )

    # -----------------------------
    # Summary
    # -----------------------------
    print(f"\n\n{'=' * 70}")
    print("DETAILED LAYER-BY-LAYER SUMMARY")
    print(f"{'=' * 70}")

    print("\n1. ASR CRDNN Encoder (encoder only):")

    # Show individual layer details if available
    if layer_names and layer_macs:
        for i, (name, macs, params) in enumerate(
            zip(layer_names, layer_macs, layer_params)
        ):
            print(f"   {chr(97 + i)}) {name}:")
            print(f"      MACs:   {macs / 1e6:>10.2f} M")
            print(f"      Params: {params / 1e6:>10.2f} M")
        print(f"   {chr(97 + len(layer_names))}) Total ASR CRDNN Encoder:")

    print(f"   Total MACs:   {macs_asr_enc / 1e6:>10.2f} M")
    print(f"   Total Params: {params_asr_enc / 1e6:>10.2f} M")

    print("\n2. SLU Encoder:")
    print(f"   a) {lstm_type} Layer ({direction_str}, {lstm_num_layers} layers):")
    print(f"      MACs:   {macs_lstm / 1e6:>10.2f} M")
    print(f"      Params: {params_lstm / 1e6:>10.2f} M")
    print("   b) Linear Layer:")
    print(f"      MACs:   {macs_linear / 1e6:>10.2f} M")
    print(f"      Params: {params_linear / 1e6:>10.2f} M")
    print("   c) Total SLU Encoder:")
    print(f"      MACs:   {macs_slu_enc / 1e6:>10.2f} M")
    print(f"      Params: {params_slu_enc / 1e6:>10.2f} M")

    print("\n3. SLU Decoder (values for last profiled sequence length):")
    print("   a) Embedding Layer:")
    print(f"      MACs:   {macs_emb / 1e6:>10.2f} M")
    print(f"      Params: {params_emb / 1e6:>10.2f} M")
    print(
        f"   b) AttentionalRNNDecoder ({dec_rnn_type} with Attention, {dec_num_layers} layers):"
    )
    print(f"      MACs:   {macs_attn_dec / 1e6:>10.2f} M")
    print(f"      Params: {params_attn_dec / 1e6:>10.2f} M")
    print("   c) Output Linear Layer:")
    print(f"      MACs:   {macs_out_lin / 1e6:>10.2f} M")
    print(f"      Params: {params_out_lin / 1e6:>10.2f} M")
    print("   d) Total Decoder:")
    print(f"      MACs:   {macs_dec / 1e6:>10.2f} M")
    print(f"      Params: {params_dec / 1e6:>10.2f} M")
    print("   Note: Decoder MACs vary with sequence length")

    print("\n4. Overall System:")
    print(
        f"   Total Encoder MACs (ASR + SLU):  {(macs_asr_enc + macs_slu_enc) / 1e6:>10.2f} M"
    )
    print(
        f"   Total Params (ASR + SLU + Dec):  {(params_asr_enc + params_slu_enc + params_dec) / 1e6:>10.2f} M"
    )

    print(f"{'=' * 70}\n")


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
