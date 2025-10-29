"""
Tokenizer training script for training a BPE tokenizer, to convert semantics into sub-word units that can
be used to train a language (LM) or an acoustic model (AM).

Usage: python train.py hparams/tokenizer.yaml
"""

import sys

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import ddp_init_group, run_on_main

from prepare import prepare_dataset

if __name__ == "__main__":
    # Command line argument parsing
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then create ddp_group with the right communication protocol
    ddp_init_group(run_opts)

    # multi-gpu (ddp) data preparation
    run_on_main(
        prepare_dataset,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["manifest_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Train tokenizer
    hparams["tokenizer"]()
