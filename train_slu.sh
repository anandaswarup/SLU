#!/usr/bin/bash

echo "Activating environment"
source .venv/bin/activate

echo "HuBERT Encoder -> Attention GRU Decoder SLU model training start"
python train_hubert_encoder_gru_decoder_slu.py --hparams hparams/hubert_encoder_gru_decoder_slu.yaml
echo "HuBERT Encoder -> Attention GRU Decoder SLU model training complete"

echo "Deactivating environment"
deactivate
