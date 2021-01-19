#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" simulate train and dev set, use multiprocessing.Pool to accelerate the pipline;
it's not totally random
@author: nwpuykjv@163.com
        arrowhyx@foxmail.com
"""


import numpy as np
import math
import soundfile as sf
import scipy.signal as sps
import librosa
import os
import warnings
import sys
eps = np.finfo(np.float32).eps
import argparse
import multiprocessing as mp
import traceback
def audioread(path, fs=16000):
    '''
    args
        path: wav path
        fs: sample rate
    return
        wave_data: L x C or L
    '''
    wave_data, sr = sf.read(path)
    if sr != fs:
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
        wave_data = librosa.resample(wave_data, sr, fs)
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
    return wave_data

def get_firstchannel_read(path, fs=16000):
    '''
    args
        path: wav path
        fs: sample rate
    return
        wave_data: L
    '''
    wave_data, sr = sf.read(path)
    if sr != fs:
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
        wave_data = librosa.resample(wave_data, sr, fs)
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
    if len(wave_data.shape) > 1:
        wave_data = wave_data[:, 0]
    return wave_data

def clip_data(data, start, segment_length):
    '''
    according the start point and segment_length to split the data
    args:
        data: numpy.array
        start: -2, -1, [0,...., L - 1]
        segment_length: int
    return:
        tgt: numpy.array
    '''
    tgt = np.zeros(segment_length)
    data_len = data.shape[0]
    if start == -2:
        """
        this means segment_length // 4 < data_len < segment_length // 2
        padding to A_A_A
        """
        if data_len < segment_length//3:
            data = np.pad(data, [0, segment_length//3 - data_len])
            tgt[:segment_length//3] += data
            st = segment_length//3
            tgt[st:st+data.shape[0]] += data
            st = segment_length//3 * 2
            tgt[st:st+data.shape[0]] += data
        
        else:
            """
            padding to A_A
            """
            # st = (segment_length//2 - data_len) % 101
            # tgt[st:st+data_len] += data
            # st = segment_length//2 + (segment_length - data_len) % 173
            # tgt[st:st+data_len] += data
            data = np.pad(data, [0, segment_length//2 - data_len])
            tgt[:segment_length//2] += data
            st = segment_length//2
            tgt[st:st+data.shape[0]] += data
    
    elif start == -1:
        '''
        this means segment_length < data_len*2
        padding to A_A
        '''
        if data_len % 4 == 0:
            tgt[:data_len] += data
            tgt[data_len:] += data[:segment_length-data_len]
        elif data_len % 4 == 1:
            tgt[:data_len] += data
        elif data_len % 4 == 2:
            tgt[-data_len:] += data
        elif data_len % 4 == 3:
            tgt[(segment_length-data_len)//2:(segment_length-data_len)//2+data_len] += data
    
    else:
        tgt += data[start:start+segment_length]
    
    return tgt

def rms(data):
    """
    calc rms of wav
    """
    energy = data ** 2
    max_e = np.max(energy)
    low_thres = max_e*(10**(-50/10)) # to filter lower than 50dB 
    rms = np.mean(energy[energy>=low_thres])
    #rms = np.mean(energy)
    return rms

def snr_mix(clean, noise, snr):
    '''
    mix clean and noise according to snr
    '''
    clean_rms = rms(clean)
    clean_rms = np.maximum(clean_rms, eps)
    noise_rms = rms(noise)
    noise_rms = np.maximum(noise_rms, eps)
    k = math.sqrt(clean_rms / (10**(snr/10) * noise_rms))
    new_noise = noise * k
    return new_noise

def mix_noise(clean, noise, snr, channels=8):
    '''
    split/pad the noise data and then mix them according to snr
    '''
    clean_length = clean.shape[0]
    noise_length = noise.shape[0]
    st = 0  # choose the first point
    # padding the noise
    if clean_length > noise_length:
        # st = numpy.random.randint(clean_length + 1 - noise_length)
        noise_t = np.zeros([clean_length, channels])
        noise_t[st:st+noise_length] = noise
        noise = noise_t
    # split the noise
    elif clean_length < noise_length:
        # st = numpy.random.randint(noise_length + 1 - clean_length)
        noise = noise[st:st+clean_length]
    
    snr_noise = snr_mix(clean, noise, snr)
    return snr_noise

def add_reverb(cln_wav, rir_wav, channels=8):
    """
    add reverberation
    args:
        cln_wav: L
        rir_wav: L x C
        rir_wav is always [Lr, C]
    return:
        wav_tgt: L x C
    """
    rir_len = rir_wav.shape[0]
    wav_tgt = np.zeros([channels, cln_wav.shape[0] + rir_len-1])
    for i in range(channels):
        wav_tgt[i] = sps.oaconvolve(cln_wav, rir_wav[:, i])
    # L x C
    wav_tgt = np.transpose(wav_tgt)
    wav_tgt = wav_tgt[:cln_wav.shape[0]]
    return wav_tgt

def get_one_spk_noise(clean, noise, snr, scale):
    """
    mix clean and noise according to the snr and scale
    args:
        clean: numpy.array, L x C  L is always segment_length
        noise: numpy.array, L' x C
        snr: float
        scale: float
    """
    gen_noise = mix_noise(clean, noise, snr)
    noisy = clean + gen_noise

    max_amp = np.max(np.abs(noisy))
    max_amp = np.maximum(max_amp, eps)
    noisy_scale = 1. / max_amp * scale
    clean = clean * noisy_scale
    noisy = noisy * noisy_scale
    return noisy, clean

def generate_data(clean_path, strat_time, noise_path, rir_path, snr, scale, segment_length=16000*4, channels=8):
    clean = get_firstchannel_read(clean_path)
    # chunk the clean wav
    clean = clip_data(clean, strat_time, segment_length)
    noise = get_firstchannel_read(noise_path)
    
    # add linear/circle rir
    rir = audioread(rir_path) 

    L, C = rir.shape

    # linear array rir is [Lr, 16]
    if C%channels == 0 and C==2*channels:
        clean_rir = rir[:, :channels]
        noise_rir = rir[:, channels:]
    elif C==channels:
        warnings.warn("the clean'rir and noise's rir will be same")
        clean_rir = rir 
        noise_rir = rir
    # circle array rir is [Lr, 32]
    #elif C%channels == 0 and C%channels == 0:
    elif C%(channels*2) == 0:
        skip = C//channels//2
        clean_rir = rir[:, :C//2:skip]   #every C//channels channels
        noise_rir = rir[:, C//2::skip]  #every C//channels channels 
        print(C,channels)
    else:
        raise RuntimeError("Can not generate target channels data, please check data or parameters")
    clean = add_reverb(clean, clean_rir, channels=channels)
    noise = add_reverb(noise, noise_rir, channels=channels)

    inputs, labels = get_one_spk_noise(clean, noise, snr, scale)
    return inputs, labels

def preprocess_func(line, segment_length, result):
    try:
        path = line.strip()
        data = get_firstchannel_read(path)
        length = data.shape[0]

        if length < segment_length:
            if length * 2 < segment_length and length * 4 > segment_length:
                result.append('{} -2\n'.format(path))
            elif length * 2 > segment_length:
                result.append('{} -1\n'.format(path))
        else:
            sample_index = 0
            while sample_index + segment_length <= length:
                result.append('{} {}\n'.format(path, sample_index))
                sample_index += segment_length
            if sample_index < length:
                result.append('{} {}\n'.format(path, length - segment_length))

    except :
        traceback.print_exc()

def get_clean_chunk(clean_path, clean_chunk_path, sample_rate=16000, chunk=4, num_process=12):
    
    '''
    split the clean_wav every chunk second
    args:
        clean_path:  
            format is   /xxx/..../yyy.wav
                        /xxy/..../zzz.wav
                        /xxy/..../aaa.wav
        clean_chunk_path: 
            format is   /xxx/..../yyy.wav -2
                        /xxy/..../zzz.wav -1
                        /xxy/..../aaa.wav [0,1...L-1]
    '''
    lines = open(clean_path, 'r').readlines()

    pool = mp.Pool(num_process)
    mgr = mp.Manager()
    result = mgr.list()
    segment_length = int(sample_rate * chunk)

    for line in lines:
        pool.apply_async(
            preprocess_func,
            args=(line, segment_length, result)
            )
    pool.close()
    pool.join()
    wid = open(clean_chunk_path, 'w')
    for item in result:
        wid.write(item)
    wid.close()

def get_mix_config(clean_chunk_path, noise_path, rir_path, config_path, snr_range=[0,30], scale_range=[0.2,0.9]):
    '''
    generate config file
        format is:  clean_path start_time noise_path rir_path snr scale
    '''
    
    clean_lines = open(clean_chunk_path, 'r').readlines()
    noise_lines = open(noise_path, 'r').readlines()
    rir_lines = open(rir_path, 'r').readlines()

    wid = open(config_path, 'w')
    noise_len = len(noise_lines)
    rir_len = len(rir_lines)

    idx = 0
    for line in clean_lines:
        clean_path = line.strip()
        noise_path = noise_lines[idx % noise_len].strip()
        rir_path = rir_lines[idx % rir_len].strip()
        snr = np.random.uniform(*snr_range)          #snr range is [0, 30)
        scale = np.random.uniform(*scale_range)     #scale range is [0.2, 0.9)
        wid.write("{} {} {} {} {}\n".format(clean_path, noise_path, rir_path, snr, scale))
        idx = idx + 1
    wid.close() 

def mix_func(line, save_dir, chunk, sample_rate):
    try:
        segment_length = int(chunk * sample_rate)
        clean_path, start_time, noise_path, rir_path, snr, scale = line.split(' ')
        # L x C
        inputs, labels = generate_data(clean_path, int(start_time), noise_path, rir_path, 
                                float(snr), float(scale), segment_length) 
        clean = os.path.basename(clean_path).replace('.wav', '')
        noise = os.path.basename(noise_path).replace('.wav', '')
        rir = os.path.basename(rir_path).replace('.wav', '')
        seg = '#'
        utt_id = clean + seg + noise + seg + rir + seg + start_time + seg + snr + seg + scale + '.wav'
        sf.write(os.path.join(save_dir, 'mix', utt_id), inputs, sample_rate)
        sf.write(os.path.join(save_dir, 'ref', utt_id), labels, sample_rate)
    except :
        traceback.print_exc()

def get_data(config_path, save_dir, chunk=4, sample_rate=16000, num_process=12):
    '''
    according to the config file to generate data and then save in save_dir
    '''
    lines = open(config_path, 'r')
    
    idx = 0
    pool = mp.Pool(num_process)
    for line in lines:
        # print('idx: ', idx)
        line = line.strip()
        # multiprocessing
        pool.apply_async(
                mix_func,
                args=(line, save_dir, chunk, sample_rate)
            )
        idx = idx + 1
    pool.close()
    pool.join()
    lines.close() 

def main(args):
    clean_path = args.clean_wav_list 
    clean_chunk_path = args.clean_wav_list+'.{}.duration'.format(args.chunk_len)

    noise_path = args.noise_wav_list 
    rir_path = args.rir_wav_list 
    config_path = args.mix_config_path
    
    save_dir = args.save_dir 
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(os.path.join(save_dir, 'mix')):
        os.mkdir(os.path.join(save_dir, 'mix'))
    if not os.path.isdir(os.path.join(save_dir, 'ref')):
        os.mkdir(os.path.join(save_dir, 'ref'))
    if args.generate_config: 
        #if not os.path.exists(clean_chunk_path):
        print('LOG: preparing clean start time')
        get_clean_chunk(clean_path, clean_chunk_path, chunk=args.chunk_len)

        print('LOG: preparing mix config')
        get_mix_config(clean_chunk_path, noise_path, rir_path, config_path)
    
    print('LOG: generating')
    get_data(config_path, save_dir, chunk=args.chunk_len)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--clean_wav_list',
        type=str,
        default='clean.lst',
        help='the list of clean wav to read'
        ) 
     
    parser.add_argument(
        '--noise_wav_list',
        type=str,
        default='noise.lst',
        help='the list of noise wav to read'
        ) 

    parser.add_argument(
        '--rir_wav_list',
        type=str,
        default='rir.lst',
        help='the list of rir wav to read'
        ) 
    
    parser.add_argument(
        '--mix_config_path',
        type=str,
        default='mix.config',
        help='the save path of config path to save'
        ) 
    
    parser.add_argument(
        '--save_dir',
        type=str,
        default='generated_data',
        help='the dir to save generated_data'
        ) 
    parser.add_argument(
        '--chunk_len',
        type=float,
        default=6,
        help='the length of one chunk sample'
        ) 
    parser.add_argument(
        '--generate_config',
        type=str,
        default='True',
        help='generate mix config file or not '
        ) 
    args = parser.parse_args()
    if args.generate_config == 'True' \
        or args.generate_config == 'true' \
        or args.generate_config == 't' \
        or args.generate_config == 'T':
        args.generate_config = True
    else:
        args.generate_config = False
    main(args)
