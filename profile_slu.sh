#!/usr/bin/bash

echo "Activating environment"
source .venv/bin/activate

echo "SLU profiling start"
python compute_macs.py --hparams hparams/crdnn_librispeech_encoder_seq2seq_slu.yaml --device cpu --compute-macs > ../results/profile_crdnn_cpu.txt
python compute_macs.py --hparams hparams/hubert_encoder_gru_decoder_slu.yaml --device cpu --compute-macs > ../results/profile_hubert_cpu.txt
echo "SLU profiling complete"

echo "Deactivating environment"
deactivate
