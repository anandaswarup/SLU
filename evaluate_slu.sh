#!/usr/bin/bash

echo "Activating environment"
source .venv/bin/activate

echo "SLU Evaluation start"

echo "Evaluating held out test set"
python evaluate_slu.py --test_csv ../data/manifests/test.csv --output_csv ../results/hubert_encoder_gru_decoder_slu/test.csv --hparams hparams/hubert_encoder_gru_decoder_slu.yaml --device cuda

echo "Evaluating unseen test set"
python evaluate_slu.py --test_csv ../data/manifests/unseen.csv --output_csv ../results/hubert_encoder_gru_decoder_slu/unseen.csv --hparams hparams/hubert_encoder_gru_decoder_slu.yaml --device cuda

echo "SLU evaluation complete"

echo "Deactivating environment"
deactivate
