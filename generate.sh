#!/usr/bin/env bash
source ./config.sh

DATA_RAW="./out/data_bin_art_1"
set -e
echo "using this file for test: "$DATA_RAW

ema='ema'
 
# this code used for split many sentences in 1 row -> many rows, each row contains a sentence
# rm -rf $DATA_RAW/test.src-tgt.src $DATA_RAW/test.src-tgt.tgt
# python gec_scripts/split.py $DATA_RAW/test.src-tgt.src.old $DATA_RAW/test.src-tgt.src $DATA_RAW/test.idx
# cp $DATA_RAW/test.src-tgt.src $DATA_RAW/test.src-tgt.tgt

epochs='_last'
for epoch in ${epochs[*]}; do
    if [ -f $RESULT/m2score$ema$exp_$epoch.log ]; then
        continue
    fi
    echo $epoch

    CUDA_VISIBLE_DEVICES=$device python generate.py $DATA_RAW \
    --path $MODELS/checkpoint$ema$epoch.pt \
    --beam 12 \
    --nbest 12 \
    --gen-subset test \
    --max-tokens 6000 \
    --no-progress-bar \
    --raw-text \
    --batch-size 128 \
    --print-alignment \
    --max-len-a 0 \
    --no-early-stop \
    --copy-ext-dict --replace-unk \
    > $RESULT/output$ema$epoch.nbest.txt 

    cat $RESULT/output$ema$epoch.nbest.txt | grep "^H" | python ./gec_scripts/sort.py 12 $RESULT/output$ema$epoch.txt.split

    python ./gec_scripts/revert_split.py $RESULT/output$ema$epoch.txt.split $DATA_RAW/test.idx > $RESULT/output$ema$epoch.txt

    python2 ./gec_scripts/m2scorer/m2scorer -v $RESULT/output$ema$epoch.txt ./data/test.m2 > $RESULT/m2score$ema$exp_$epoch.log
    tail -n 1 $RESULT/m2score$ema$exp_$epoch.log
done

python gec_scripts/show_m2.py $RESULT/m2score$ema$exp_{}.log
