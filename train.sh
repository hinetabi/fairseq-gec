#!/bin/bash
DATA='./data' # input dir
OUT='./out' # output dir

DATA_BIN=$OUT/data_bin
DATA_RAW=$OUT/data_raw
mkdir -p $DATA_BIN
mkdir -p $DATA_RAW

MODELS=$OUT/models$exp
RESULT=$OUT/result$exp
mkdir -p $MODELS
mkdir -p $RESULT

# source ./config.sh

# pretrained_model=./out/models_pretrain/checkpoint9.pt
exp=first
CUDA_VISIBLE_DEVICES=0 nohup python train.py $DATA_BIN \
--save-dir $MODELS \
--source-lang src \
--target-lang tgt \
--seed 4321 \
--max-epoch 15 \
--batch-size 64 \
--max-tokens 3000 \
--train-subset train \
--valid-subset valid \
--arch transformer \
--lr-scheduler triangular --max-lr 0.004 --lr-period-updates 73328 \
--clip-norm 2 --lr 0.001 --lr-shrink 0.95 --shrink-min \
--dropout 0.2 --relu-dropout 0.2 --attention-dropout 0.2 --copy-attention-dropout 0.2 \
--encoder-embed-dim 512 --decoder-embed-dim 512 \
--max-target-positions 1024 --max-source-positions 1024 \
--encoder-ffn-embed-dim 4096 --decoder-ffn-embed-dim 4096 \
--encoder-attention-heads 8 --decoder-attention-heads 8 \
--copy-attention-heads 1 \
--share-all-embeddings \
--no-progress-bar \
--log-interval 1000 \
--positive-label-weight 1.2 \
--copy-attention --copy-attention-heads 1 > $OUT/log$exp.out 2>&1 &

tail -f $OUT/log$exp.out
