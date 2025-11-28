"""
Compute MACs and latency for SLU models. This script profiles both CRDNN and HuBERT-based SLU models, computing MACs
(multiply-accumulate operations) and inference latency.
"""

import argparse
import sys
import time
from typing import Any

import torch
import torch.nn as nn
from hyperpyyaml import load_hyperpyyaml
from ptflops import get_model_complexity_info


def profile_model(
    model: nn.Module,
    input_constructor: Any,
    verbose: bool = False,
    custom_modules_hooks: dict | None = None,
) -> tuple[float, float]:
    """Profile a model using ptflops.

    Args:
        model: The model to profile.
        input_constructor: Function to construct input for the model.
        verbose: Whether to print verbose output.
        custom_modules_hooks: Custom hooks for specific module types.

    Returns:
        Tuple of (MACs, params) in billions/millions.
    """
    macs, params = get_model_complexity_info(
        model,
        input_res=(1,),
        input_constructor=input_constructor,
        as_strings=False,
        print_per_layer_stat=verbose,
        custom_modules_hooks=custom_modules_hooks or {},
    )
    return float(macs), float(params)  # type: ignore[arg-type]


def get_model_type(hparams: dict) -> str:
    """Determine the model type from hyperparameters.

    Args:
        hparams: The loaded hyperparameters dictionary.

    Returns:
        Either "crdnn" or "hubert" based on the config.
    """
    if "asr_model_path" in hparams:
        return "crdnn"
    elif "hubert_hub" in hparams or "hubert" in hparams:
        return "hubert"
    else:
        raise ValueError(
            "Unknown model type. Expected 'asr_model_path' for CRDNN "
            "or 'hubert_hub'/'hubert' for HuBERT in hparams."
        )


# =============================================================================
# Model Wrappers for CRDNN
# =============================================================================


class ASREncoderCNNWrapper(nn.Module):
    """Wrapper for ASR encoder CNN layers."""

    def __init__(self, asr_encoder: nn.Module) -> None:
        super().__init__()
        self.cnn = asr_encoder.CNN  # type: ignore[union-attr]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)  # type: ignore[misc]


class ASREncoderRNNWrapper(nn.Module):
    """Wrapper for ASR encoder RNN layers."""

    def __init__(self, asr_encoder: nn.Module) -> None:
        super().__init__()
        self.rnn = asr_encoder.RNN  # type: ignore[union-attr]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rnn(x)  # type: ignore[misc]


class ASREncoderDNNWrapper(nn.Module):
    """Wrapper for ASR encoder DNN layers."""

    def __init__(self, asr_encoder: nn.Module) -> None:
        super().__init__()
        self.dnn = asr_encoder.DNN  # type: ignore[union-attr]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dnn(x)  # type: ignore[misc]


class ASREncoderWrapper(nn.Module):
    """Wrapper for full ASR encoder."""

    def __init__(self, asr_encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = asr_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# =============================================================================
# Model Wrappers for HuBERT
# =============================================================================


class HuBERTEncoderWrapper(nn.Module):
    """Wrapper for HuBERT encoder."""

    def __init__(self, hubert: nn.Module) -> None:
        super().__init__()
        self.hubert = hubert

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hubert(x)


# =============================================================================
# Model Wrappers for SLU Components (shared)
# =============================================================================


class SLUEncoderLSTMWrapper(nn.Module):
    """Wrapper for SLU encoder LSTM."""

    def __init__(self, slu_encoder: nn.Module) -> None:
        super().__init__()
        self.lstm = slu_encoder.lstm  # type: ignore[union-attr]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lstm(x)[0]  # type: ignore[misc]


class SLUEncoderLinearWrapper(nn.Module):
    """Wrapper for SLU encoder linear layer."""

    def __init__(self, slu_encoder: nn.Module) -> None:
        super().__init__()
        self.linear = slu_encoder.linear  # type: ignore[union-attr]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)  # type: ignore[misc]


class SLUEncoderWrapper(nn.Module):
    """Wrapper for full SLU encoder."""

    def __init__(self, slu_encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = slu_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)[0]


class SLUDecoderEmbeddingWrapper(nn.Module):
    """Wrapper for SLU decoder embedding layer."""

    def __init__(self, slu_decoder: nn.Module) -> None:
        super().__init__()
        self.embedding = slu_decoder.emb  # type: ignore[union-attr]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)  # type: ignore[misc]


class SLUDecoderAttentionalRNNWrapper(nn.Module):
    """Wrapper for SLU decoder attentional RNN."""

    def __init__(self, slu_decoder: nn.Module) -> None:
        super().__init__()
        self.rnn = slu_decoder.dec  # type: ignore[union-attr]

    def forward(
        self,
        enc_states: torch.Tensor,
        enc_lens: torch.Tensor,
        embedded: torch.Tensor,
    ) -> torch.Tensor:
        return self.rnn(enc_states, enc_lens, embedded)  # type: ignore[misc]


class SLUDecoderLinearWrapper(nn.Module):
    """Wrapper for SLU decoder linear layer."""

    def __init__(self, slu_decoder: nn.Module) -> None:
        super().__init__()
        self.linear = slu_decoder.fc  # type: ignore[union-attr]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)  # type: ignore[misc]


class SLUDecoderWrapper(nn.Module):
    """Wrapper for full SLU decoder (embedding + attention + linear)."""

    def __init__(self, slu_decoder: nn.Module) -> None:
        super().__init__()
        self.decoder = slu_decoder

    def forward(
        self,
        enc_states: torch.Tensor,
        enc_lens: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.decoder.emb(tokens)  # type: ignore[union-attr]
        dec_out, _ = self.decoder.dec(enc_states, enc_lens, emb)  # type: ignore[union-attr]
        return self.decoder.fc(dec_out)  # type: ignore[union-attr]


# =============================================================================
# MAC Computation for CRDNN
# =============================================================================


def compute_macs_crdnn(hparams: dict, device: str, verbose: bool = False) -> dict:
    """Compute MACs for CRDNN-based SLU model.

    Args:
        hparams: The loaded hyperparameters.
        device: Device to run profiling on.
        verbose: Whether to print detailed stats.

    Returns:
        Dictionary with MAC counts for each component.
    """
    # Load ASR encoder
    asr_encoder = hparams["asr_model"]
    asr_encoder.to(device)
    asr_encoder.eval()

    # SLU components
    slu_encoder = hparams["slu_encoder"]
    slu_encoder.to(device)
    slu_encoder.eval()

    slu_decoder = hparams["seq_decoder"]
    slu_decoder.to(device)
    slu_decoder.eval()

    # Input dimensions
    input_length = 16000 * 10  # 10 seconds of audio
    hop_length = hparams.get("hop_length", 160)
    n_mels = hparams.get("n_mels", 80)
    seq_length = (input_length // hop_length) + 1
    output_tokens = 50  # Average output sequence length

    results = {}

    # Profile ASR encoder CNN
    def cnn_input_constructor(input_res: tuple) -> dict:
        return {"x": torch.randn(1, seq_length, n_mels).to(device)}

    cnn_wrapper = ASREncoderCNNWrapper(asr_encoder)
    cnn_wrapper.to(device)
    cnn_macs, cnn_params = profile_model(
        cnn_wrapper, cnn_input_constructor, verbose=verbose
    )
    results["asr_cnn"] = {"macs": cnn_macs, "params": cnn_params}

    # Get CNN output shape for RNN input
    with torch.no_grad():
        cnn_out = cnn_wrapper(torch.randn(1, seq_length, n_mels).to(device))
        cnn_out_shape = cnn_out.shape

    # Profile ASR encoder RNN
    def rnn_input_constructor(input_res: tuple) -> dict:
        return {"x": torch.randn(cnn_out_shape).to(device)}

    rnn_wrapper = ASREncoderRNNWrapper(asr_encoder)
    rnn_wrapper.to(device)
    rnn_macs, rnn_params = profile_model(
        rnn_wrapper, rnn_input_constructor, verbose=verbose
    )
    results["asr_rnn"] = {"macs": rnn_macs, "params": rnn_params}

    # Get RNN output shape for DNN input
    with torch.no_grad():
        rnn_out = rnn_wrapper(torch.randn(cnn_out_shape).to(device))
        rnn_out_shape = rnn_out.shape

    # Profile ASR encoder DNN
    def dnn_input_constructor(input_res: tuple) -> dict:
        return {"x": torch.randn(rnn_out_shape).to(device)}

    dnn_wrapper = ASREncoderDNNWrapper(asr_encoder)
    dnn_wrapper.to(device)
    dnn_macs, dnn_params = profile_model(
        dnn_wrapper, dnn_input_constructor, verbose=verbose
    )
    results["asr_dnn"] = {"macs": dnn_macs, "params": dnn_params}

    # Get full ASR encoder output shape
    with torch.no_grad():
        asr_out = asr_encoder(torch.randn(1, seq_length, n_mels).to(device))
        asr_out_shape = asr_out.shape

    # Profile SLU encoder LSTM
    def slu_lstm_input_constructor(input_res: tuple) -> dict:
        return {"x": torch.randn(asr_out_shape).to(device)}

    slu_lstm_wrapper = SLUEncoderLSTMWrapper(slu_encoder)
    slu_lstm_wrapper.to(device)
    slu_lstm_macs, slu_lstm_params = profile_model(
        slu_lstm_wrapper, slu_lstm_input_constructor, verbose=verbose
    )
    results["slu_encoder_lstm"] = {"macs": slu_lstm_macs, "params": slu_lstm_params}

    # Get LSTM output shape
    with torch.no_grad():
        lstm_out = slu_lstm_wrapper(torch.randn(asr_out_shape).to(device))
        lstm_out_shape = lstm_out.shape

    # Profile SLU encoder linear
    def slu_linear_input_constructor(input_res: tuple) -> dict:
        return {"x": torch.randn(lstm_out_shape).to(device)}

    slu_linear_wrapper = SLUEncoderLinearWrapper(slu_encoder)
    slu_linear_wrapper.to(device)
    slu_linear_macs, slu_linear_params = profile_model(
        slu_linear_wrapper, slu_linear_input_constructor, verbose=verbose
    )
    results["slu_encoder_linear"] = {
        "macs": slu_linear_macs,
        "params": slu_linear_params,
    }

    # Get SLU encoder output shape
    with torch.no_grad():
        slu_enc_out = slu_encoder(torch.randn(asr_out_shape).to(device))[0]
        slu_enc_out_shape = slu_enc_out.shape

    # Profile SLU decoder embedding
    def emb_input_constructor(input_res: tuple) -> dict:
        return {"x": torch.randint(0, 1000, (1, output_tokens)).to(device)}

    emb_wrapper = SLUDecoderEmbeddingWrapper(slu_decoder)
    emb_wrapper.to(device)
    emb_macs, emb_params = profile_model(
        emb_wrapper, emb_input_constructor, verbose=verbose
    )
    results["slu_decoder_embedding"] = {"macs": emb_macs, "params": emb_params}

    # Profile SLU decoder attention RNN
    def attn_input_constructor(input_res: tuple) -> dict:
        enc_states = torch.randn(slu_enc_out_shape).to(device)
        enc_lens = torch.ones(1).to(device)
        emb_out = torch.randn(1, output_tokens, slu_decoder.emb.embedding_dim).to(
            device
        )
        return {"enc_states": enc_states, "enc_lens": enc_lens, "embedded": emb_out}

    attn_wrapper = SLUDecoderAttentionalRNNWrapper(slu_decoder)
    attn_wrapper.to(device)
    attn_macs, attn_params = profile_model(
        attn_wrapper, attn_input_constructor, verbose=verbose
    )
    results["slu_decoder_attention"] = {"macs": attn_macs, "params": attn_params}

    # Profile SLU decoder linear
    dec_hidden_size = slu_decoder.dec.rnn.hidden_size

    def dec_linear_input_constructor(input_res: tuple) -> dict:
        return {"x": torch.randn(1, output_tokens, dec_hidden_size).to(device)}

    dec_linear_wrapper = SLUDecoderLinearWrapper(slu_decoder)
    dec_linear_wrapper.to(device)
    dec_linear_macs, dec_linear_params = profile_model(
        dec_linear_wrapper, dec_linear_input_constructor, verbose=verbose
    )
    results["slu_decoder_linear"] = {
        "macs": dec_linear_macs,
        "params": dec_linear_params,
    }

    # Compute totals
    total_macs = sum(r["macs"] for r in results.values())
    total_params = sum(r["params"] for r in results.values())
    results["total"] = {"macs": total_macs, "params": total_params}

    return results


# =============================================================================
# MAC Computation for HuBERT
# =============================================================================


def compute_macs_hubert(hparams: dict, device: str, verbose: bool = False) -> dict:
    """Compute MACs for HuBERT-based SLU model.

    The HuBERT model consists of:
    - HuBERT encoder
    - Attentional GRU decoder (embedding + attention RNN + linear)

    Args:
        hparams: The loaded hyperparameters.
        device: Device to run profiling on.
        verbose: Whether to print detailed stats.

    Returns:
        Dictionary with MAC counts for each component.
    """
    # Load HuBERT encoder
    hubert = hparams["hubert"]
    hubert.to(device)
    hubert.eval()

    # Decoder
    slu_decoder = hparams["seq_decoder"]
    slu_decoder.to(device)
    slu_decoder.eval()

    # Input dimensions
    input_length = 16000 * 10  # 10 seconds of audio
    output_tokens = 50  # Average output sequence length

    results = {}

    # Profile HuBERT encoder
    def hubert_input_constructor(input_res: tuple) -> dict:
        return {"x": torch.randn(1, input_length).to(device)}

    hubert_wrapper = HuBERTEncoderWrapper(hubert)
    hubert_wrapper.to(device)
    hubert_macs, hubert_params = profile_model(
        hubert_wrapper, hubert_input_constructor, verbose=verbose
    )
    results["hubert_encoder"] = {"macs": hubert_macs, "params": hubert_params}

    # Get HuBERT output shape for decoder input
    with torch.no_grad():
        hubert_out = hubert(torch.randn(1, input_length).to(device))
        hubert_out_shape = hubert_out.shape

    # Profile decoder embedding
    def emb_input_constructor(input_res: tuple) -> dict:
        return {"x": torch.randint(0, 1000, (1, output_tokens)).to(device)}

    emb_wrapper = SLUDecoderEmbeddingWrapper(slu_decoder)
    emb_wrapper.to(device)
    emb_macs, emb_params = profile_model(
        emb_wrapper, emb_input_constructor, verbose=verbose
    )
    results["decoder_embedding"] = {"macs": emb_macs, "params": emb_params}

    # Profile decoder attention RNN
    def attn_input_constructor(input_res: tuple) -> dict:
        enc_states = torch.randn(hubert_out_shape).to(device)
        enc_lens = torch.ones(1).to(device)
        emb_out = torch.randn(1, output_tokens, slu_decoder.emb.embedding_dim).to(
            device
        )
        return {"enc_states": enc_states, "enc_lens": enc_lens, "embedded": emb_out}

    attn_wrapper = SLUDecoderAttentionalRNNWrapper(slu_decoder)
    attn_wrapper.to(device)
    attn_macs, attn_params = profile_model(
        attn_wrapper, attn_input_constructor, verbose=verbose
    )
    results["decoder_attention"] = {"macs": attn_macs, "params": attn_params}

    # Profile decoder linear
    dec_hidden_size = slu_decoder.dec.rnn.hidden_size

    def dec_linear_input_constructor(input_res: tuple) -> dict:
        return {"x": torch.randn(1, output_tokens, dec_hidden_size).to(device)}

    dec_linear_wrapper = SLUDecoderLinearWrapper(slu_decoder)
    dec_linear_wrapper.to(device)
    dec_linear_macs, dec_linear_params = profile_model(
        dec_linear_wrapper, dec_linear_input_constructor, verbose=verbose
    )
    results["decoder_linear"] = {
        "macs": dec_linear_macs,
        "params": dec_linear_params,
    }

    # Compute totals
    total_macs = sum(r["macs"] for r in results.values())
    total_params = sum(r["params"] for r in results.values())
    results["total"] = {"macs": total_macs, "params": total_params}

    return results


# =============================================================================
# Latency Computation for CRDNN
# =============================================================================


def compute_latency_crdnn(
    hparams: dict, device: str, num_runs: int = 100, warmup_runs: int = 10
) -> dict:
    """Compute inference latency for CRDNN-based SLU model.

    Args:
        hparams: The loaded hyperparameters.
        device: Device to run profiling on.
        num_runs: Number of profiling runs.
        warmup_runs: Number of warmup runs.

    Returns:
        Dictionary with latency stats for each component.
    """
    # Load models
    asr_encoder = hparams["asr_model"]
    asr_encoder.to(device)
    asr_encoder.eval()

    slu_encoder = hparams["slu_encoder"]
    slu_encoder.to(device)
    slu_encoder.eval()

    slu_decoder = hparams["seq_decoder"]
    slu_decoder.to(device)
    slu_decoder.eval()

    # Input dimensions
    input_length = 16000 * 10  # 10 seconds
    hop_length = hparams.get("hop_length", 160)
    n_mels = hparams.get("n_mels", 80)
    seq_length = (input_length // hop_length) + 1
    output_tokens = 50

    results = {}

    # Create inputs
    mel_input = torch.randn(1, seq_length, n_mels).to(device)

    # Initialize asr_out for later use
    asr_out = asr_encoder(mel_input)

    # Warmup and measure ASR encoder
    with torch.no_grad():
        for _ in range(warmup_runs):
            asr_encoder(mel_input)

        if device == "cuda":
            torch.cuda.synchronize()

        asr_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            asr_out = asr_encoder(mel_input)
            if device == "cuda":
                torch.cuda.synchronize()
            asr_times.append(time.perf_counter() - start)

    results["asr_encoder"] = {
        "mean_ms": sum(asr_times) / len(asr_times) * 1000,
        "std_ms": (
            sum((t - sum(asr_times) / len(asr_times)) ** 2 for t in asr_times)
            / len(asr_times)
        )
        ** 0.5
        * 1000,
    }

    # Measure SLU encoder
    # Initialize slu_enc_out for later use
    slu_enc_out, _ = slu_encoder(asr_out)

    with torch.no_grad():
        for _ in range(warmup_runs):
            slu_encoder(asr_out)

        if device == "cuda":
            torch.cuda.synchronize()

        slu_enc_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            slu_enc_out, _ = slu_encoder(asr_out)
            if device == "cuda":
                torch.cuda.synchronize()
            slu_enc_times.append(time.perf_counter() - start)

    results["slu_encoder"] = {
        "mean_ms": sum(slu_enc_times) / len(slu_enc_times) * 1000,
        "std_ms": (
            sum(
                (t - sum(slu_enc_times) / len(slu_enc_times)) ** 2
                for t in slu_enc_times
            )
            / len(slu_enc_times)
        )
        ** 0.5
        * 1000,
    }

    # Measure SLU decoder
    enc_lens = torch.ones(1).to(device)
    tokens = torch.randint(0, 1000, (1, output_tokens)).to(device)

    decoder_wrapper = SLUDecoderWrapper(slu_decoder)
    decoder_wrapper.to(device)
    decoder_wrapper.eval()

    with torch.no_grad():
        for _ in range(warmup_runs):
            decoder_wrapper(slu_enc_out, enc_lens, tokens)

        if device == "cuda":
            torch.cuda.synchronize()

        dec_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            decoder_wrapper(slu_enc_out, enc_lens, tokens)
            if device == "cuda":
                torch.cuda.synchronize()
            dec_times.append(time.perf_counter() - start)

    results["slu_decoder"] = {
        "mean_ms": sum(dec_times) / len(dec_times) * 1000,
        "std_ms": (
            sum((t - sum(dec_times) / len(dec_times)) ** 2 for t in dec_times)
            / len(dec_times)
        )
        ** 0.5
        * 1000,
    }

    # Total latency
    results["total"] = {
        "mean_ms": results["asr_encoder"]["mean_ms"]
        + results["slu_encoder"]["mean_ms"]
        + results["slu_decoder"]["mean_ms"],
        "std_ms": (
            results["asr_encoder"]["std_ms"] ** 2
            + results["slu_encoder"]["std_ms"] ** 2
            + results["slu_decoder"]["std_ms"] ** 2
        )
        ** 0.5,
    }

    return results


# =============================================================================
# Latency Computation for HuBERT
# =============================================================================


def compute_latency_hubert(
    hparams: dict, device: str, num_runs: int = 100, warmup_runs: int = 10
) -> dict:
    """Compute inference latency for HuBERT-based SLU model.

    The HuBERT model consists of:
    - HuBERT encoder
    - Attentional GRU decoder

    Args:
        hparams: The loaded hyperparameters.
        device: Device to run profiling on.
        num_runs: Number of profiling runs.
        warmup_runs: Number of warmup runs.

    Returns:
        Dictionary with latency stats for each component.
    """
    # Load models
    hubert = hparams["hubert"]
    hubert.to(device)
    hubert.eval()

    slu_decoder = hparams["seq_decoder"]
    slu_decoder.to(device)
    slu_decoder.eval()

    # Input dimensions
    input_length = 16000 * 10  # 10 seconds
    output_tokens = 50

    results = {}

    # Create inputs
    audio_input = torch.randn(1, input_length).to(device)

    # Initialize hubert_out for later use
    hubert_out = hubert(audio_input)

    # Warmup and measure HuBERT encoder
    with torch.no_grad():
        for _ in range(warmup_runs):
            hubert(audio_input)

        if device == "cuda":
            torch.cuda.synchronize()

        hubert_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            hubert_out = hubert(audio_input)
            if device == "cuda":
                torch.cuda.synchronize()
            hubert_times.append(time.perf_counter() - start)

    results["hubert_encoder"] = {
        "mean_ms": sum(hubert_times) / len(hubert_times) * 1000,
        "std_ms": (
            sum((t - sum(hubert_times) / len(hubert_times)) ** 2 for t in hubert_times)
            / len(hubert_times)
        )
        ** 0.5
        * 1000,
    }

    # Measure decoder
    enc_lens = torch.ones(1).to(device)
    tokens = torch.randint(0, 1000, (1, output_tokens)).to(device)

    decoder_wrapper = SLUDecoderWrapper(slu_decoder)
    decoder_wrapper.to(device)
    decoder_wrapper.eval()

    with torch.no_grad():
        for _ in range(warmup_runs):
            decoder_wrapper(hubert_out, enc_lens, tokens)

        if device == "cuda":
            torch.cuda.synchronize()

        dec_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            decoder_wrapper(hubert_out, enc_lens, tokens)
            if device == "cuda":
                torch.cuda.synchronize()
            dec_times.append(time.perf_counter() - start)

    results["decoder"] = {
        "mean_ms": sum(dec_times) / len(dec_times) * 1000,
        "std_ms": (
            sum((t - sum(dec_times) / len(dec_times)) ** 2 for t in dec_times)
            / len(dec_times)
        )
        ** 0.5
        * 1000,
    }

    # Total latency
    results["total"] = {
        "mean_ms": results["hubert_encoder"]["mean_ms"] + results["decoder"]["mean_ms"],
        "std_ms": (
            results["hubert_encoder"]["std_ms"] ** 2 + results["decoder"]["std_ms"] ** 2
        )
        ** 0.5,
    }

    return results


# =============================================================================
# Result Formatting
# =============================================================================


def format_macs_results(results: dict, model_type: str) -> str:
    """Format MAC computation results for display.

    Args:
        results: Dictionary of MAC results.
        model_type: Type of model ("crdnn" or "hubert").

    Returns:
        Formatted string.
    """
    lines = [f"\n{'=' * 60}", f"MACs Profile for {model_type.upper()} Model", "=" * 60]

    for name, data in results.items():
        if name == "total":
            continue
        macs_g = data["macs"] / 1e9
        params_m = data["params"] / 1e6
        lines.append(f"{name:30s}: {macs_g:8.3f} GMACs, {params_m:8.3f} M params")

    lines.append("-" * 60)
    total_macs_g = results["total"]["macs"] / 1e9
    total_params_m = results["total"]["params"] / 1e6
    lines.append(
        f"{'TOTAL':30s}: {total_macs_g:8.3f} GMACs, {total_params_m:8.3f} M params"
    )
    lines.append("=" * 60)

    return "\n".join(lines)


def format_latency_results(results: dict, model_type: str) -> str:
    """Format latency results for display.

    Args:
        results: Dictionary of latency results.
        model_type: Type of model ("crdnn" or "hubert").

    Returns:
        Formatted string.
    """
    lines = [
        f"\n{'=' * 60}",
        f"Latency Profile for {model_type.upper()} Model",
        "=" * 60,
    ]

    for name, data in results.items():
        if name == "total":
            continue
        lines.append(f"{name:30s}: {data['mean_ms']:8.2f} ± {data['std_ms']:.2f} ms")

    lines.append("-" * 60)
    lines.append(
        f"{'TOTAL':30s}: {results['total']['mean_ms']:8.2f} "
        f"± {results['total']['std_ms']:.2f} ms"
    )
    lines.append("=" * 60)

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute MACs and latency for SLU models."
    )
    parser.add_argument(
        "--hparams",
        type=str,
        required=True,
        help="Path to the hyperparameters YAML file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run profiling on (default: cpu).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-layer statistics.",
    )
    parser.add_argument(
        "--compute-macs",
        action="store_true",
        dest="compute_macs",
        help="Compute MACs for the model.",
    )
    parser.add_argument(
        "--compute-latency",
        action="store_true",
        dest="compute_latency",
        help="Compute inference latency for the model.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        dest="num_runs",
        help="Number of runs for latency profiling (default: 100).",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=10,
        dest="warmup_runs",
        help="Number of warmup runs for latency profiling (default: 10).",
    )

    args = parser.parse_args()

    # Load hyperparameters
    with open(args.hparams, "r") as f:
        hparams = load_hyperpyyaml(f)

    # Determine model type
    model_type = get_model_type(hparams)
    print(f"Detected model type: {model_type.upper()}")

    # Default to computing both if neither specified
    if not args.compute_macs and not args.compute_latency:
        args.compute_macs = True
        args.compute_latency = True

    # Compute MACs
    if args.compute_macs:
        print("\nComputing MACs...")
        if model_type == "crdnn":
            macs_results = compute_macs_crdnn(hparams, args.device, args.verbose)
        else:
            macs_results = compute_macs_hubert(hparams, args.device, args.verbose)
        print(format_macs_results(macs_results, model_type))

    # Compute latency
    if args.compute_latency:
        print("\nComputing latency...")
        if model_type == "crdnn":
            latency_results = compute_latency_crdnn(
                hparams, args.device, args.num_runs, args.warmup_runs
            )
        else:
            latency_results = compute_latency_hubert(
                hparams, args.device, args.num_runs, args.warmup_runs
            )
        print(format_latency_results(latency_results, model_type))

    return 0


if __name__ == "__main__":
    sys.exit(main())
