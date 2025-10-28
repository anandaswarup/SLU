# Spoken language understanding (SLU)

## Overview
This repository contains code for training a spoken language understanding (SLU) model to predict `action`, `slot_name` and `slot_value` from audio using [SpeechBrain](https://github.com/speechbrain/speechbrain/tree/develop).


## Dataset Format
The dataset should contain three CSV files: `train.csv`, `valid.csv`, and `test.csv`. Each CSV file should have the following columns:
- `path`: Path to the audio file.
- `transcription`: The transcription of the audio.
- `action`: The action label.
- `slot_name`: The name of the slot.
- `slot_value`: The value of the slot.