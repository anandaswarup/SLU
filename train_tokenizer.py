"""
Script for training a BPE tokenizer. The tokenizer converts semantics into sub-word units that can be used to train a
Spoken language Understanding (SLU) model.

Usage: python train_tokenizer.py --hparams hparams/tokenizer.yaml
"""

import argparse
from pathlib import Path

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import ddp_init_group, run_on_main

from prepare import prepare_SLU_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer for Spoken Language Understanding"
    )
    parser.add_argument(
        "--hparams",
        type=str,
        required=True,
        help="Path to the hyperparameters YAML file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for training (default: cuda:0)",
    )
    parser.add_argument(
        "--data_parallel_backend",
        action="store_true",
        help="Enable data parallel backend for multi-GPU training",
    )
    parser.add_argument(
        "--distributed_launch",
        action="store_true",
        help="Enable distributed launch for multi-node training",
    )
    parser.add_argument(
        "--distributed_backend",
        type=str,
        default="nccl",
        help="Backend for distributed training (default: nccl)",
    )
    parser.add_argument(
        "--overrides",
        type=str,
        default="",
        help="YAML overrides for hyperparameters",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Build run_opts dictionary for SpeechBrain
    run_opts = {
        "device": args.device,
        "data_parallel_backend": args.data_parallel_backend,
        "distributed_launch": args.distributed_launch,
        "distributed_backend": args.distributed_backend,
    }

    # Read and load hyperparameters file with command-line overrides
    with open(args.hparams, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, args.overrides)

    # create ddp_group with the right communication protocol
    ddp_init_group(run_opts)

    # Create experiment directory for logging artifacts
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=None,
        overrides=args.overrides,
    )

    script_copy = Path(hparams["output_folder"]) / Path(__file__).name
    if script_copy.exists():
        script_copy.unlink()

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_SLU_dataset,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["manifest_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Train tokenizer
    hparams["tokenizer"]()
