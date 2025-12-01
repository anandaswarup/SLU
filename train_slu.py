"""
Training script for Spoken Language Understanding (SLU) models.

Supports multiple SLU architectures:
- CRDNN ASR Encoder (pre-trained on LibriSpeech) -> LSTM Encoder -> Attention GRU Decoder
- HuBERT Encoder -> Attention GRU Decoder

The model architecture is determined by the hyperparameters YAML file.

Usage:
    python train_slu.py --hparams hparams/crdnn_librispeech_encoder_seq2seq_slu.yaml
    python train_slu.py --hparams hparams/distillHuBERT_encoder_gru_decoder_slu.yaml
"""

import argparse
from pathlib import Path

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import ddp_init_group, if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

from prepare import prepare_SLU_dataset

logger = get_logger(__name__)

# Global variables set from args
show_results_every = 100
tokenizer = None


class BaseSLU(sb.Brain):
    """
    Base class for SLU models with common functionality.
    """

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
                tokenizer.decode_ids(utt_seq).split(" ")  # type: ignore
                for utt_seq in predicted_tokens
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
        Gets called at the beginning of each epoch.
        """
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):  # type: ignore
        """
        Gets called at the end of an epoch.
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
            self._on_valid_stage_end(stage_stats, epoch)
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_wer_file, "w", encoding="utf-8") as w:
                    self.wer_metric.write_stats(w)

    def _on_valid_stage_end(self, stage_stats, epoch):
        """
        Handle validation stage end. Override in subclasses for custom LR scheduling.
        """
        old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
        sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)  # type: ignore
        self.hparams.train_logger.log_stats(
            stats_meta={"epoch": epoch, "lr": old_lr},
            train_stats=self.train_stats,
            valid_stats=stage_stats,
        )
        self.checkpointer.save_and_keep_only(  # type: ignore
            meta={"WER": stage_stats["WER"]},
            min_keys=["WER"],
        )


class CRDNNSLU(BaseSLU):
    """
    SLU model using pre-trained CRDNN ASR Encoder -> LSTM Encoder -> Attention GRU Decoder.
    """

    def compute_forward(self, batch, stage):
        """
        Forward computations from the waveform batches to the output probabilities.
        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, tokens_bos_lens = batch.tokens_bos

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            orig_device = wavs.device
            wavs_cpu = wavs.to("cpu")
            wav_lens_cpu = wav_lens.to("cpu")

            with torch.no_grad():
                aug_wavs_cpu, aug_wav_lens_cpu = self.hparams.wav_augment(
                    wavs_cpu, wav_lens_cpu
                )

            wavs = aug_wavs_cpu.to(orig_device)
            wav_lens = aug_wav_lens_cpu.to(orig_device)

            tokens_bos = self.hparams.wav_augment.replicate_labels(tokens_bos)
            tokens_bos_lens = self.hparams.wav_augment.replicate_labels(tokens_bos_lens)

        # ASR encoder forward pass
        with torch.no_grad():
            ASR_encoder_out = self.hparams.asr_model.encode_batch(
                wavs.detach(), wav_lens
            )

        # SLU forward pass
        encoder_out = self.hparams.slu_enc(ASR_encoder_out)
        e_in = self.hparams.output_emb(tokens_bos)
        h, _ = self.hparams.dec(e_in, encoder_out, wav_lens)

        # Output layer for seq2seq log-probabilities
        logits = self.hparams.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        if stage == sb.Stage.TRAIN and self.step % show_results_every != 0:
            return p_seq, wav_lens
        else:
            p_tokens, _, _, _ = self.hparams.beam_searcher(encoder_out, wav_lens)
            return p_seq, wav_lens, p_tokens


class HuBERTSLU(BaseSLU):
    """
    SLU model using HuBERT Encoder -> Attention GRU Decoder.
    """

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

        # Encoder forward pass
        hubert_out = self.modules.hubert(wavs, wav_lens)  # type: ignore

        # SLU forward pass
        e_in = self.hparams.output_emb(tokens_bos)
        h, _ = self.hparams.dec(e_in, hubert_out, wav_lens)

        # Output layer for seq2seq log-probabilities
        logits = self.hparams.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        if stage == sb.Stage.TRAIN and self.step % show_results_every != 0:
            return p_seq, wav_lens
        else:
            hyps, _, _, _ = self.hparams.beam_searcher(hubert_out.detach(), wav_lens)
            return p_seq, wav_lens, hyps

    def _on_valid_stage_end(self, stage_stats, epoch):
        """
        Handle validation stage end with dual LR schedulers for HuBERT.
        """
        old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
        old_lr_hubert, new_lr_hubert = self.hparams.lr_annealing_hubert(
            stage_stats["WER"]
        )
        sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)  # type: ignore
        sb.nnet.schedulers.update_learning_rate(  # type: ignore
            self.hubert_optimizer, new_lr_hubert
        )
        self.hparams.train_logger.log_stats(
            stats_meta={
                "epoch": epoch,
                "lr": old_lr,
                "hubert_lr": old_lr_hubert,
            },
            train_stats=self.train_stats,
            valid_stats=stage_stats,
        )
        self.checkpointer.save_and_keep_only(  # type: ignore
            meta={"WER": stage_stats["WER"]},
            min_keys=["WER"],
        )

    def init_optimizers(self):
        """
        Initializes the HuBERT optimizer and model optimizer.
        """
        self.hubert_optimizer = self.hparams.hubert_opt_class(
            self.modules.hubert.parameters()  # type: ignore
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("hubert_opt", self.hubert_optimizer)
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

        self.optimizers_dict = {
            "hubert_optimizer": self.hubert_optimizer,
            "model_optimizer": self.optimizer,
        }


def dataio_prepare(hparams):
    """
    Prepares the datasets to be used in the brain class.
    Defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(  # type: ignore
        csv_path=hparams["csv_train"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        train_data = train_data.filtered_sorted(sort_key="duration")
        hparams["dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(sort_key="duration", reverse=True)
        hparams["dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "random":
        pass
    else:
        raise NotImplementedError("sorting must be random, ascending or descending")

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(  # type: ignore
        csv_path=hparams["csv_dev"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(  # type: ignore
        csv_path=hparams["csv_test"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    tok = hparams["tokenizer"]

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")  # type: ignore
    @sb.utils.data_pipeline.provides("sig")  # type: ignore
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)  # type: ignore
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)  # type: ignore

    # Define text pipeline
    @sb.utils.data_pipeline.takes("semantics")  # type: ignore
    @sb.utils.data_pipeline.provides(  # type: ignore
        "semantics", "token_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(semantics):
        yield semantics
        tokens_list = tok.encode_as_ids(semantics)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + tokens_list)
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)  # type: ignore

    # Set output keys
    sb.dataio.dataset.set_output_keys(  # type: ignore
        datasets,
        ["id", "sig", "semantics", "tokens_bos", "tokens_eos", "tokens"],
    )

    return train_data, valid_data, test_data, tok


def get_model_type(hparams):
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


def setup_crdnn_model(hparams, run_opts):
    """
    Sets up the CRDNN-based SLU model.
    """
    from speechbrain.inference.ASR import EncoderDecoderASR

    # Prepare augmentation data if available
    if "prepare_noise_data" in hparams:
        run_on_main(hparams["prepare_noise_data"])
    if "prepare_rir_data" in hparams:
        run_on_main(hparams["prepare_rir_data"])

    # Load the pre-trained ASR model
    hparams["asr_model"] = EncoderDecoderASR.from_hparams(
        source=hparams["asr_model_path"],
        run_opts={"device": run_opts["device"]},
    )

    return CRDNNSLU


def setup_hubert_model(hparams, run_opts):
    """
    Sets up the HuBERT-based SLU model.
    """
    # Move HuBERT to device
    hparams["hubert"] = hparams["hubert"].to(run_opts["device"])

    # Freeze the feature extractor part when unfreezing
    if not hparams["freeze_hubert"] and hparams["freeze_hubert_conv"]:
        hparams["hubert"].model.feature_extractor._freeze_parameters()

    return HuBERTSLU


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Spoken Language Understanding (SLU) model"
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


def main():
    global show_results_every, tokenizer

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

    # Create ddp_group with the right communication protocol
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

    # Prepare SLU dataset
    run_on_main(
        prepare_SLU_dataset,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["manifest_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Prepare datasets and tokenizer
    train_set, valid_set, test_set, tokenizer = dataio_prepare(hparams)

    # Load pretrained tokenizer
    hparams["pretrainer"].collect_files()
    hparams["pretrainer"].load_collected()

    # Determine model type and set up accordingly
    model_type = get_model_type(hparams)
    logger.info(f"Detected model type: {model_type}")

    if model_type == "crdnn":
        SLUClass = setup_crdnn_model(hparams, run_opts)
    else:  # hubert
        SLUClass = setup_hubert_model(hparams, run_opts)

    # Brain class initialization
    slu_brain = SLUClass(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Add tokenizer to trainer
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
    slu_brain.evaluate(test_set, test_loader_kwargs=hparams["dataloader_opts"])


if __name__ == "__main__":
    main()
