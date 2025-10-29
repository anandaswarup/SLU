"""
Dataset preparation script for Spoken Language Understanding (SLU) tasks.
This script reads CSV files with the following columns:
    - path: Path to the audio file.
    - transcription: The transcription of the audio.
    - action: The action label.
    - slot_name: The name of the slot.
    - slot_value: The value of the slot.
and creates manifest files for train, dev, and test splits.
"""

import logging
import os

import pandas as pd
from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)


def prepare_dataset(data_folder, save_folder, skip_prep=False):
    """
    This method prepares the dataset by creating manifest files for train, dev and test splits

    Args:
        data_folder : path to data folder containing the `train.csv`, `dev.csv`, and `test.csv` files.
        save_folder: folder where the manifest files will be stored.
        skip_prep: If True, skip data preparation.
    """
    if skip_prep:
        return

    splits = [
        "train",
        "dev",
        "test",
    ]

    # Counter for unique IDs for each audio
    ID_start = 0

    # Create save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    for split in splits:
        new_filename = os.path.join(save_folder, split) + ".csv"
        if os.path.exists(new_filename):
            continue
        logger.info("Preparing %s..." % new_filename)

        ID = []
        duration = []

        wav = []
        wav_format = []
        wav_opts = []

        semantics = []
        semantics_format = []
        semantics_opts = []

        transcript = []
        transcript_format = []
        transcript_opts = []

        df = pd.read_csv(os.path.join(data_folder, split) + ".csv")
        for i in range(len(df)):
            ID.append(ID_start + i)
            signal = read_audio(df.path[i])
            duration.append(signal.shape[0] / 16000)

            wav.append(df.path[i])
            wav_format.append("wav")
            wav_opts.append(None)

            transcript_ = df.transcription[i]
            transcript.append(transcript_)
            transcript_format.append("string")
            transcript_opts.append(None)

            semantics_ = f"{{action: {df.action[i]} | slot_name: {df.slot_name[i]} | slot_value: {df.slot_value[i]}}}"
            semantics.append(semantics_)
            semantics_format.append("string")
            semantics_opts.append(None)

        new_df = pd.DataFrame(
            {
                "ID": ID,
                "duration": duration,
                "wav": wav,
                "wav_format": wav_format,
                "wav_opts": wav_opts,
                "semantics": semantics,
                "semantics_format": semantics_format,
                "semantics_opts": semantics_opts,
                "transcript": transcript,
                "transcript_format": transcript_format,
                "transcript_opts": transcript_opts,
            }
        )
        new_df.to_csv(new_filename, index=False)
        ID_start += len(df)
