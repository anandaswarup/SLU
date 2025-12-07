# Spoken Language Understanding (SLU)

## Overview

This repository contains code for training a spoken language understanding (SLU) model to predict a semantic string containing `scenario`, `action` and `entities` from audio using [SpeechBrain](https://github.com/speechbrain/speechbrain/tree/develop).

### Supported Architectures

- **CRDNN + LSTM + Attention GRU Decoder**: Uses a pre-trained ASR encoder (trained on LibriSpeech) with an LSTM encoder and attention-based GRU decoder
- **distillHuBERT + Attention GRU Decoder**: Uses a pre-trained distillHuBERT encoder with an attention-based GRU decoder

## Installation

### Prerequisites

- Python >= 3.12
- CUDA-compatible GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/anandaswarup/slu.git
cd slu
```

2. Create a virtual environment and install dependencies using [uv](https://github.com/astral-sh/uv):
```bash
uv venv
source .venv/bin/activate
uv sync
```

## Data Preparation

### Dataset Format

Your dataset should be in JSONL format with the following structure:
```json
{"audio_filepath": "path/to/audio.wav", "duration": 2.5, "semantics": {"scenario": "...", "action": "...", "entities": [...]}, "transcript": "optional transcription"}
```

Organize your data files as:
```
data/
└── jsonl/
    ├── train.jsonl
    ├── dev.jsonl
    └── test.jsonl
```

The `prepare.py` script will automatically convert JSONL files to CSV manifests required for training.

## Training

### Step 1: Train the Tokenizer

Before training the SLU model, you need to train a tokenizer on your semantic labels:

```bash
python train_tokenizer.py --hparams hparams/tokenizer.yaml
```

**Key tokenizer parameters** (in `hparams/tokenizer.yaml`):
- `token_type`: Tokenization type (`unigram`, `bpe`, or `char`)
- `token_output`: Vocabulary size
- `data_folder`: Path to JSONL data directory
- `output_folder`: Path to save the trained tokenizer

### Step 2: Train the SLU Model

Choose one of the available architectures:

#### Option A: distillHuBERT Encoder

```bash
python train_slu.py --hparams hparams/distillHuBERT_encoder_gru_decoder_slu.yaml
```

#### Option B: CRDNN + LSTM Encoder (with LibriSpeech pre-training)

```bash
python train_slu.py --hparams hparams/crdnn_librispeech_encoder_seq2seq_slu.yaml
```

### Training Options

```bash
python train_slu.py --hparams <path_to_hparams.yaml> \
    --device cuda:0 \
    --data_parallel_backend \
    --distributed_launch \
    --overrides "batch_size=16, lr=0.0003"
```

**Common hyperparameters** (configurable in YAML files):
| Parameter | Description |
|-----------|-------------|
| `number_of_epochs` | Number of training epochs |
| `batch_size` | Training batch size |
| `lr` | Learning rate |
| `output_folder` | Directory for model checkpoints |
| `data_folder` | Path to JSONL data |
| `tokenizer_file` | Path to trained tokenizer model |

## Evaluation

Evaluate a trained model on a test set:

```bash
python evaluate_slu.py \
    --test_csv data/manifests/test.csv \
    --output_csv results/predictions.csv \
    --hparams hparams/distillHuBERT_encoder_gru_decoder_slu.yaml \
    --device cuda
```

**Arguments:**
- `--test_csv`: Path to the test manifest CSV file
- `--output_csv`: Path to save prediction results
- `--hparams`: Path to the hyperparameters YAML file
- `--device`: Device to use (`cpu` or `cuda`)

## Inference

Run inference on a single audio file:

```bash
python inference_slu.py \
    --wav_file path/to/audio.wav \
    --hparams hparams/distillHuBERT_encoder_gru_decoder_slu.yaml \
    --device cuda
```

**Arguments:**
- `--wav_file`: Path to the input audio file (WAV format, will be resampled to 16kHz if needed)
- `--hparams`: Path to the hyperparameters YAML file
- `--device`: Device to use (`cpu` or `cuda`)

**Output:** The script outputs the predicted semantic string containing scenario, action, and entities.

## Project Structure

```
slu/
├── train_slu.py           # Main training script
├── train_tokenizer.py     # Tokenizer training script
├── inference_slu.py       # Single-file inference script
├── evaluate_slu.py        # Batch evaluation script
├── prepare.py             # Data preparation utilities
├── compute_macs.py        # Model complexity analysis
├── hparams/
│   ├── tokenizer.yaml                              # Tokenizer configuration
│   ├── distillHuBERT_encoder_gru_decoder_slu.yaml  # distillHuBERT model config
│   └── crdnn_librispeech_encoder_seq2seq_slu.yaml  # CRDNN model config
├── data/
│   ├── jsonl/             # Raw JSONL data files
│   └── manifests/         # Generated CSV manifests
└── models/                # Trained model checkpoints
```

## License

See [LICENSE](LICENSE) for details.