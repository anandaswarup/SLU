"""
Script for training a BPE tokenizer. The tokenizer converts semantics into sub-word units that can be used to train a
Spoken language Understanding (SLU) model.

Usage: python train.py hparams/tokenizer.yaml
"""

import sys
from pathlib import Path

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import ddp_init_group, run_on_main

from prepare import prepare_SLU_dataset

if __name__ == "__main__":
    # Parse command-line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Read and load hyperparameters file with command-line overrides
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    ddp_init_group(run_opts)

    model_dir = Path(hparams["output_folder"])
    log_dir = Path(
        hparams.get("log_folder", model_dir.parent / "logs" / model_dir.name)
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create experiment directory for logging artifacts
    sb.create_experiment_directory(
        experiment_directory=str(log_dir),
        hyperparams_to_save=None,
        overrides=overrides,
    )

    script_copy = log_dir / Path(sys.argv[0]).name
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
