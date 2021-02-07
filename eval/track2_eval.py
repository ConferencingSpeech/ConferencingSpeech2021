'''

for eval the model, pesq, stoi, si-snr

need to install pypesq: 
https://github.com/ludlows/python-pesq

pystoi:
https://github.com/mpariente/pystoi

'''

import soundfile as sf
import multiprocessing as mp
import argparse
import numpy as np 
import os
os.environ['OMP_NUM_THREADS'] = '2'

def audioread(path, fs=16000):
    wave_data, sr = sf.read(path)
    assert fs == sr
    if len(wave_data.shape) >= 2:
        wave_data = wave_data[:,0]

    return wave_data, fs

def rms(data):
    """
    calc rms of wav
    """
    energy = data ** 2
    max_e = np.max(energy)
    low_thres = max_e * (10**(-50/10)) # to filter lower than 50dB 
    rms = np.mean(energy[energy >= low_thres])
    #rms = np.mean(energy)
    return rms

def snr(enh, noisy, eps=1e-8):
    noise = enh - noisy
    return 10 * np.log10((rms(enh) + eps) / (rms(noise) + eps))

def eval(enh_name, nsy_name, kind, results):
    try:
        utt_id = enh_name.split('/')[-1]
        enh, sr = audioread(enh_name)
        nsy, sr = audioread(nsy_name)
        enh_len = enh.shape[0]
        nsy_len = nsy.shape[0]
        if enh_len > nsy_len:
            enh = enh[:nsy_len]
        else:
            nsy = nsy[:enh_len]
        enh_sisdr = snr(enh, nsy)

    except Exception as e:
        print(e)
    
    results.append([kind, utt_id, enh_sisdr])

def main(args):
    enh_dir = args.enh_dir
    mix_dir = args.mix_dir

    pool = mp.Pool(args.num_threads)
    mgr = mp.Manager()
    results = mgr.list()
    with open(args.write_path, 'w') as wfid:
        for item in ['simu_circular', 'simu_linear_nonuniform', 'simu_linear_uniform']:
            pathe = enh_dir.format(item)
            pathn = mix_dir.format(item)
            files = os.listdir(pathe)
            for file in files:
                pool.apply_async(
                    eval,
                    args=(
                        os.path.join(pathe,file),
                        os.path.join(pathn,file),
                        item,
                        results,
                    )
                )
        pool.close()
        pool.join()
        dic = dict()
        
        for eval_score in results:
            kind, utt_id, score = eval_score
            if utt_id not in dic.keys():
                dic[utt_id] = dict()
            dic[utt_id][kind] = score
        for utt in dic.keys():
            maxm = -100000
            max_kind = ""
            for kind in dic[utt].keys():
                if dic[utt][kind] > maxm:
                    maxm = dic[utt][kind]
                    max_kind = kind
            wfid.write('{} {} {}\n'.format(utt, max_kind, maxm))

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--write_path',
        type=str,
        default='write_path'
        ) 
    parser.add_argument(
        '--num_threads',
        type=int,
        default=16
        )
    parser.add_argument(
        '--enh_dir',
        type=str,
        default='enh_dir'
        )
    parser.add_argument(
        '--mix_dir',
        type=str,
        default='mix_dir'
        )
    args = parser.parse_args()
    main(args)
