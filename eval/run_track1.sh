#!/bin/bash

enh='/path/track1'

for dir in linear_uniform circular linear_nonuniform ; do
    ls ${enh}/${dir} > ./tmp/track1_${dir}
    python track1_eval.py \
        --wav_list=./tmp/track1_${dir} \
        --pathe=${enh}/${dir} \
        --pathc=/data/wavs/dev/dev_simu_${dir}_track1/reverb_ref/ \
        --pathn=/data/wavs/dev/dev_simu_${dir}_track1/mix/ \
        --result_list=./result/track1_${dir}_result.csv
done
