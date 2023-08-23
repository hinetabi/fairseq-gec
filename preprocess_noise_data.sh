#!/usr/bin/env bash
source ./config.sh

set -e
set -x

# set copy params
copy_params='--copy-ext-dict'

# set common params between train/test
common_params='--source-lang src --target-lang tgt  
--padding-factor 1 
--srcdict ./dicts/dict.src.txt 
--joined-dictionary 
'

for epoch in {1..1}; do
    echo $epoch

    # testpref=$DATA/test_1b_$epoch
    trainpref=$DATA/train_1b_$epoch

    # validpref=$DATA/valid

    # add valid if process valid data
    # --validpref $validpref \


    # preprocess train/valid
    python preprocess.py \
    $common_params \
    $copy_params \
    --trainpref $trainpref \
    --destdir ${DATA_BIN}_art_$epoch \
    --output-format raw \
    --alignfile $trainpref.forward \
    | tee $OUT/data_bin_art_$epoch.log
done
 