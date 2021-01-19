#!/bin/bash


enh='/home/work_nfs4_ssd/yxhu/workspace/snr_est/exp/se_snr_cldnn_sigmoid_more/tencent/'
#for dir in linear ; do
for dir in circle linear ; do
    
    ls ../data/wavs/dev/simu_${dir}/ref/ > /tmp/score_${dir}
    python eval_objective.py \
        --wav_list=/tmp/score_${dir}\
        --pathe=${enh}\
        --pathc=../data/wavs/dev/simu_${dir}/ref/ \
        --pathn=../data/wavs/dev/simu_${dir}/mix/ \
        --result_list=result_${dir}.csv

done
