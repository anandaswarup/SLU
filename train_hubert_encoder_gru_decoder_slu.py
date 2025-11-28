"""
Script for training a HuBERT/wav2vec2 Encoder -> Attention GRU Decoder SLU model with NLL loss.

Usage: python train_hubert_encoder_gru_decoder_slu.py --hparams hparams/hubert_encoder_gru_decoder_slu.yaml
"""

import argparse
from pathlib import Path

import pandas as pd
import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import ddp_init_group, if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

from prepare import prepare_SLU_dataset

logger = get_logger(__name__)


class SLU(sb.Brain):
    def compute_forward(self, batch, stage):
        """
        Forward computations from the waveform batches to the output probabilities.
        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, tokens_bos_lens = batch.tokens_bos

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
            tokens_bos = self.hparams.wav_augment.replicate_labels(tokens_bos)

        #  encoder forward pass
        wav2vec2_out = self.modules.wav2vec2(wavs, wav_lens)  # type: ignore

        # SLU forward pass
        e_in = self.hparams.output_emb(tokens_bos)
        h, _ = self.hparams.dec(e_in, wav2vec2_out, wav_lens)

        # Output layer for seq2seq log-probabilities
        logits = self.hparams.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        if stage == sb.Stage.TRAIN and self.step % show_results_every != 0:
            return p_seq, wav_lens
        else:
            hyps, _, _, _ = self.hparams.beam_searcher(wav2vec2_out.detach(), wav_lens)

            return p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """
        Computes the loss (NLL) given predictions and targets.
        """
        predicted_tokens = None
        if stage == sb.Stage.TRAIN and self.step % show_results_every != 0:
            p_seq, wav_lens = predictions
        else:
            p_seq, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        # Label Augmentation
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            tokens_eos = self.hparams.wav_augment.replicate_labels(tokens_eos)
            tokens_eos_lens = self.hparams.wav_augment.replicate_labels(tokens_eos_lens)

        loss_seq = self.hparams.seq_cost(p_seq, tokens_eos, length=tokens_eos_lens)

        loss = loss_seq

        if (
            (stage != sb.Stage.TRAIN) or (self.step % show_results_every == 0)
        ) and predicted_tokens is not None:
            # Decode token terms to words
            predicted_semantics = [
                tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in predicted_tokens
            ]

            target_semantics = [wrd.split(" ") for wrd in batch.semantics]

            for i in range(len(target_semantics)):
                print(" ".join(predicted_semantics[i]).replace("|", ","))
                print(" ".join(target_semantics[i]).replace("|", ","))
                print("")

            if stage != sb.Stage.TRAIN:
                self.wer_metric.append(ids, predicted_semantics, target_semantics)
                self.cer_metric.append(ids, predicted_semantics, target_semantics)

        return loss

    def on_stage_start(self, stage, epoch):  # type: ignore
        """
        Gets called at the beginning of each epoch
        """

        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):  # type: ignore
        """
        Gets called at the end of a epoch.
        """
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            (
                old_lr_wav2vec2,
                new_lr_wav2vec2,
            ) = self.hparams.lr_annealing_wav2vec2(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)  # type: ignore
            sb.nnet.schedulers.update_learning_rate(  # type: ignore
                self.wav2vec2_optimizer, new_lr_wav2vec2
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr": old_lr,
                    "wave2vec_lr": old_lr_wav2vec2,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(  # type: ignore
                meta={"WER": stage_stats["WER"]},
                min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_wer_file, "w", encoding="utf-8") as w:
                    self.wer_metric.write_stats(w)

    def init_optimizers(self):
        """
        Initializes the wav2vec2 optimizer and model optimizer
        """
        self.wav2vec2_optimizer = self.hparams.wav2vec2_opt_class(
            self.modules.wav2vec2.parameters()  # type: ignore
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("wav2vec2_opt", self.wav2vec2_optimizer)
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

        self.optimizers_dict = {
            "wav2vec_optimizer": self.wav2vec2_optimizer,
            "model_optimizer": self.optimizer,
        }


def dataio_prepare(hparams):
    """
    This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(  # type: ignore
        csv_path=hparams["csv_train"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(sort_key="duration", reverse=True)
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError("sorting must be random, ascending or descending")

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(  # type: ignore
        csv_path=hparams["csv_valid"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(  # type: ignore
        csv_path=hparams["csv_test"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")  # type: ignore
    @sb.utils.data_pipeline.provides("sig")  # type: ignore
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)  # type: ignore
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)  # type: ignore

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("semantics")  # type: ignore
    @sb.utils.data_pipeline.provides(  # type: ignore
        "semantics", "token_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(semantics):
        yield semantics
        tokens_list = tokenizer.encode_as_ids(semantics)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)  # type: ignore

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(  # type: ignore
        datasets,
        ["id", "sig", "semantics", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_data, tokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a HuBERT Encoder -> Attention GRU Decoder SLU model"
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
    parser.add_argument(
        "--show_results_every",
        type=int,
        default=100,
        help="Show results every N iterations (default: 100)",
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

    # Load hyperparameters file with command-line overrides
    with open(args.hparams, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, args.overrides)

    show_results_every = args.show_results_every

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

    # here we create the datasets objects as well as tokenization and encoding
    (train_set, valid_set, test_set, tokenizer) = dataio_prepare(hparams)

    # We download and pretrain the tokenizer
    hparams["pretrainer"].collect_files()
    hparams["pretrainer"].load_collected()

    # Move the wav2vec2
    hparams["wav2vec2"] = hparams["wav2vec2"].to(run_opts["device"])

    # freeze the feature extractor part when unfreezing
    if not hparams["freeze_wav2vec2"] and hparams["freeze_wav2vec2_conv"]:
        hparams["wav2vec2"].model.feature_extractor._freeze_parameters()

    # Brain class initialization
    slu_brain = SLU(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    slu_brain.tokenizer = tokenizer  # type: ignore

    # Training
    slu_brain.fit(
        slu_brain.hparams.epoch_counter,
        train_set,
        valid_set,
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )

    # Test
    print("Creating id_to_file mapping...")
    id_to_file = {}
    df = pd.read_csv(hparams["csv_test"])
    for i in range(len(df)):
        id_to_file[str(df.ID[i])] = df.wav[i].split("/")[-1]

    slu_brain.evaluate(test_set, test_loader_kwargs=hparams["dataloader_opts"])
