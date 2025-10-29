"""
Script for training a tokenizer on the semantics strings of the dataset. The tokenizer converts semantic strings
into tokens that can be used to train a language (LM) or an acoustic model (AM).

Usage: python train_tokenizer.py hparams/tokenizer.yaml
"""

import sys

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import ddp_init_group, run_on_main

from prepare import prepare_dataset

if __name__ == "__main__":
    # Parse command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command line overrides
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_dataset,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Train tokenizer
    hparams["tokenizer"]()
