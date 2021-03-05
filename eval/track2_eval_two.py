'''

for eval the model, pesq, stoi, estio, si-sdr

need to install pypesq: 
https://github.com/ludlows/python-pesq

pystoi:
https://github.com/mpariente/pystoi


'''

import soundfile as sf
from pesq import pesq
import multiprocessing as mp
import argparse
from pystoi.stoi import stoi
import numpy as np 
import os
os.environ['OMP_NUM_THREADS'] = '2'

def audioread(path, fs=16000):
    wave_data, sr = sf.read(path)
    assert fs == sr
    if len(wave_data.shape) >= 2:
        wave_data = wave_data[:,0]

    return wave_data, fs

def remove_dc(signal):
    """Normalized to zero mean"""
    mean = np.mean(signal)
    signal -= mean
    return signal


def pow_np_norm(signal):
    """Compute 2 Norm"""
    return np.square(np.linalg.norm(signal, ord=2))


def pow_norm(s1, s2):
    return np.sum(s1 * s2)


def si_snr(estimated, original, eps=1e-8):
    # estimated = remove_dc(estimated)
    # original = remove_dc(original)
    target = pow_norm(estimated, original) * original / pow_np_norm(original)
    noise = estimated - target
    return 10 * np.log10((pow_np_norm(target) + eps) / (pow_np_norm(noise) + eps))

def eval(ref_format, enh_name, nsy_format, results):
    try:
        utt_id = enh_name.split('/')[-1]
        enh, sr = audioread(enh_name)

        mix_score = -10000
        mix_stoi = -10000
        mix_estoi = -10000
        mix_sisdr = -10000
        final_score = -10000
        final_stoi = -10000
        final_estoi = -10000
        final_sisdr = -10000
        for kind in ['circular', 'linear_nonuniform', 'linear_uniform']:
            ref_name = ref_format.format(kind)
            nsy_name = nsy_format.format(kind)
            ref, sr = audioread(ref_name)
            nsy, sr = audioread(nsy_name)
            enh_len = enh.shape[0]
            ref_len = ref.shape[0]
            if enh_len > ref_len:
                enh = enh[:ref_len]
            else:
                ref = ref[:enh_len]
                nsy = nsy[:enh_len]
            ref_score = pesq(sr,ref, nsy, 'wb')
            enh_score = pesq(sr,ref, enh, 'wb')
            ref_stoi = stoi(ref, nsy, sr, extended=False)
            enh_stoi = stoi(ref, enh, sr, extended=False)
            ref_estoi = stoi(ref, nsy, sr, extended=True)
            enh_estoi = stoi(ref, enh, sr, extended=True)
            ref_sisdr = si_snr(nsy, ref)
            enh_sisdr = si_snr(enh, ref)
            if enh_score > final_score:
                mix_score = ref_score
                final_score = enh_score
            if enh_stoi > final_stoi:
                mix_stoi = ref_stoi
                final_stoi = enh_stoi
            if enh_estoi > final_estoi:
                mix_estoi = ref_estoi
                final_estoi = enh_estoi
            if enh_sisdr > final_sisdr:
                mix_sisdr = ref_sisdr
                final_sisdr = enh_sisdr

    except Exception as e:
        print(e)
    
    results.append([utt_id, 
                    {'pesq':[mix_score, final_score],
                     'stoi':[mix_stoi, final_stoi],
                     'estoi':[mix_estoi, final_estoi],
                     'si_sdr':[mix_sisdr, final_sisdr],
                    }])

def main(args):
    enh_dir = args.enh_dir
    ref_dir = args.ref_dir
    mix_dir = args.mix_dir
    pool = mp.Pool(args.num_threads)
    mgr = mp.Manager()
    results = mgr.list()
    lines = open(args.read_path).readlines()
    with open(args.result_list, 'w') as wfid:
        for line in lines:
            line = line.strip()
            file, kind, score = line.split()
            pool.apply_async(
                eval,
                args=(
                    os.path.join(ref_dir,file),
                    os.path.join(enh_dir.format(kind),file),
                    os.path.join(mix_dir,file),
                    results,
                )
            )
        pool.close()
        pool.join()
        for eval_score in results:
            utt_id, score = eval_score
            pesq = score['pesq']
            stoi = score['stoi']
            estoi = score['estoi']
            si_sdr = score['si_sdr']
            wfid.writelines(
                    '{:s},{:.3f},{:.3f}, '.format(utt_id, pesq[0],pesq[1])
                )
            wfid.writelines(
                    '{:.3f},{:.3f}, '.format(stoi[0],stoi[1])
                )
            wfid.writelines(
                    '{:.3f},{:.3f}, '.format(estoi[0],estoi[1])
                )
            wfid.writelines(
                    '{:.3f},{:.3f}\n'.format(si_sdr[0],si_sdr[1])
                )


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--read_path',
        type=str,
        default='read_path'
        ) 
    parser.add_argument(
        '--result_list',
        type=str,
        default='result_list'
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
        '--ref_dir',
        type=str,
        default='ref_dir'
        )
    parser.add_argument(
        '--mix_dir',
        type=str,
        default='mix_dir'
        )
    args = parser.parse_args()
    main(args)
