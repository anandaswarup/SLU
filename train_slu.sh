#!/usr/bin/bash

echo "Activating environment"
source .venv/bin/activate

echo "SLU model training start"
python train_slu.py --hparams hparams/hubert_encoder_gru_decoder_slu.yaml
echo "SLU model training complete"

echo "Deactivating environment"
deactivate
