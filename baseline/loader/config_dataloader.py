import numpy as np
import numpy
import math
import soundfile as sf
import scipy.signal as sps
import librosa
import random
import os

import torch
import torch as th

import torch.utils.data as tud
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp 

eps = np.finfo(np.float32).eps

def audioread(path, fs=16000):
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
            data = np.pad(data, [0, segment_length//3 - data_len], 'constant')
            tgt[:segment_length//3] += data
            st = segment_length//3
            tgt[st:st+data.shape[0]] += data
            st = segment_length//3 * 2
            tgt[st:st+data.shape[0]] += data
        
        else:
            """
            padding to A_A
            """
            data = np.pad(data, [0, segment_length//2 - data_len], 'constant')
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
    elif C%(channels*2) == 0:
        skip = C//channels//2
        clean_rir = rir[:, :C//2:skip]   #every C//channels channels
        noise_rir = rir[:, C//2::skip]  #every C//channels channels
    else:
        raise RuntimeError("Can not generate target channels data, please check data or parameters")
    clean = add_reverb(clean, clean_rir, channels=channels)
    noise = add_reverb(noise, noise_rir, channels=channels)

    inputs, labels = get_one_spk_noise(clean, noise, snr, scale)
    return inputs, labels


def parse_scp(scp, path_list):
    with open(scp) as fid:
        for line in fid:
            tmp = line.strip()
            path_list.append(tmp)

class Config_Dataset(Dataset):

    def __init__(   self,
                    config_scp,
                    repeat=1,
                    chunk=4,
                    sample_rate=16000,
                ):
        super(Config_Dataset, self).__init__()

        self.wav_list = list()
        parse_scp(config_scp, self.wav_list)
        self.segment_length = chunk * sample_rate
        self.wav_list *= repeat
    
    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, index):

        line = self.wav_list[index]
        line = line.split()
        assert len(line) == 6
        clean_path, start_time, noise_path, rir_path, snr, scale = line

        mix, ref = generate_data(clean_path, int(start_time), noise_path, rir_path, 
                                float(snr), float(scale), self.segment_length) 

        assert(mix.shape[0] == self.segment_length)
        assert(ref.shape[0] == self.segment_length)

        mix = mix.transpose((1, 0)).astype(np.float32)
        ref = ref.transpose((1, 0)).astype(np.float32)

        egs = {
            "mix": mix,     # C x L
            "ref": ref,     # C x L
        }
        return egs

def make_config_loader(config_scp, batch_size=8, repeat=1, num_workers=16, 
                        chunk=4, sample_rate=16000):

    dataset = Config_Dataset(
                config_scp=config_scp,
                repeat=repeat,
                chunk=chunk,
                sample_rate=sample_rate,
            )

    loader = tud.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=False,
                shuffle=True,
            )
    return loader

def test_loader():

    config_scp = '../wav_list/tr/linear_100.lst'
    repeat = 1
    num_worker = 16
    chunk = 4
    sample_rate = 16000
    batch_size = 3

    loader = make_config_loader(
                config_scp=config_scp,
                batch_size=batch_size,
                repeat=repeat,
                num_workers=num_worker,
                chunk=chunk,
                sample_rate=sample_rate,
            )
    
    cnt = 0
    print('len: ', len(loader))
    for idx, egs in enumerate(loader):
        cnt = cnt + 1
        print('cnt: {}'.format(cnt))
        print('egs["mix"].shape: ', egs["mix"].shape)
        print('egs["ref"].shape: ', egs["ref"].shape)
        print()
        # sf.write('./data/input/inputs_' + str(cnt) + '.wav', egs["mix"][0], 16000)
        # sf.write('./data/label/inputs_' + str(cnt) + '.wav', egs["ref"][0], 16000)
        if cnt >= 10:
            break
    print('done!')

if __name__ == "__main__":
    test_loader()
