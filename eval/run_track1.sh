#!/bin/bash

enh='/home/work_nfs4_ssd/ykjv/data/tencent_track1'

for dir in linear circle non_uniform ; do
    ls ${enh}/${dir} > ./tmp/${dir}
    python track1_eval.py \
        --wav_list=./tmp/${dir} \
        --pathe=${enh}/${dir} \
        --pathc=/home/work_nfs4_ssd/yxhu/data/tencent/simulate_script/data/wavs/dev/simu_${dir}/ref/ \
        --pathn=/home/work_nfs4_ssd/yxhu/data/tencent/simulate_script/data/wavs/dev/simu_${dir}/mix/ \
        --result_list=./result/track1_${dir}_result.csv
done
