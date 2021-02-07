#!/bin/bash

tmp_path='./tmp/tmp_track2'
enh_dir='/data/track2/new_{}/'
ref_dir='/data/simu_task2/dev_task2/{}/ref/'
mix_dir='/data/simu_task2/dev_task2/{}/mix/'
result_list='./result/track2_result.csv'

for dir in simu_circular simu_linear_nonuniform simu_linear_uniform ; do
    python track2_eval.py \
        --write_path=${tmp_path} \
        --enh_dir=${enh_dir} \
        --mix_dir=${mix_dir}
done

python track2_eval_two.py \
    --read_path=${tmp_path} \
    --result_list=${result_list} \
    --enh_dir=${enh_dir} \
    --ref_dir=${ref_dir} \
    --mix_dir=${mix_dir}
