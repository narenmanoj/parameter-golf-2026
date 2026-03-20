#!/bin/bash
# Combined run: int6 + MLP3x + SmearGate + MTP + NorMuon + QAT + weight sharing
# + sliding window eval + overtone init + phase-transition residual mixing
# + FP16 tied embeddings + Muon WD + AdamW

RUN_ID=combined_int6_mlp3x \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=4200 \
NUM_LAYERS=18 \
NUM_UNIQUE_LAYERS=9 \
MLP_MULT=3 \
QUANT_BITS=6 \
QAT_START_FRAC=0.75 \
QAT_LR_MULT=0.5 \
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.1 \
NORMUON_BETA=0.95 \
MUON_WEIGHT_DECAY=0.02 \
ADAM_WEIGHT_DECAY=0.01 \
SLIDING_WINDOW_STRIDE=64 \
torchrun --standalone --nproc_per_node=1 train_gpt_combined.py
