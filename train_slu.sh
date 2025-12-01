#!/usr/bin/bash

echo "Activating environment"
source .venv/bin/activate

echo "SLU model training start"
python python train_slu.py --hparams hparams/distillHuBERT_encoder_gru_decoder_slu.yaml
echo "SLU model training complete"

echo "Deactivating environment"
deactivate
