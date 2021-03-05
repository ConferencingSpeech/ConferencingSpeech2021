#!/bin/bash

tmp_path='./tmp/tmp_track2'
enh_dir='/path/track2/{}/'
ref_dir='/data/wavs/dev/dev_simu_{}_track2/reverb_ref/'
mix_dir='/data/wavs/dev/dev_simu_{}_track2/mix/'
result_list='./result/track2_result.csv'

python track2_eval.py \
    --write_path=${tmp_path} \
    --enh_dir=${enh_dir} \
    --mix_dir=${mix_dir}

python track2_eval_two.py \
    --read_path=${tmp_path} \
    --result_list=${result_list} \
    --enh_dir=${enh_dir} \
    --ref_dir=${ref_dir} \
    --mix_dir=${mix_dir}