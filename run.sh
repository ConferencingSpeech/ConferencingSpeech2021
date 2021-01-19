#!/bin/bash

clean_wav_path='../dev/clean.scp'
noise_wave_path='../dev/noise.scp'
rir_wave_path='../dev/selected/'

for item in circle linear ; do
    save_dir=./data/wavs/dev/simu_${item}
    write_path=data/dev_simu_${item}_mix.config
    python mix_wav.py  \
        --clean_wav_list=${clean_wav_path} \
        --noise_wav_list=${noise_wave_path} \
        --rir_wav_list=${rir_wave_path}/${item}.scp \
        --mix_config_path=${write_path} \
        --save_dir=${save_dir}  \
        --chunk_len=6 \
        --generate_config=True 
done

