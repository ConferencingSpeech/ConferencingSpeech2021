import numpy as np
import soundfile as sf
import librosa
import torch
import torch as th

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

def parse_scp(scp, path_list):
    with open(scp) as fid:
        for line in fid:
            tmp = line.strip().split()
            if len(tmp) > 1:
                path_list.append({'inputs': tmp[0], 'duration': float(tmp[1])})
            else:
                path_list.append({'inputs': tmp[0]})

class DataReader(object):
    def __init__(self, file_name, sample_rate=16000, kind=None):
        self.file_list = []
        self.kind = kind
        parse_scp(file_name, self.file_list)

    def extract_feature(self, path):
        path = path['inputs']
        utt_id = path.split('/')[-1].replace(".wav", "")
        data = audioread(path).astype(np.float32)

        if data.shape[-1] == 16:
            if self.kind == 'circle':
                data = data[:, ::2]
            elif self.kind == 'linear':
                data = data[:, :8]

        # C x L
        inputs = data.transpose((1, 0))
        inputs = np.reshape(inputs, [1, inputs.shape[0], inputs.shape[1]])
        inputs = torch.from_numpy(inputs)
        egs = {
            'mix': inputs,
            'utt_id': utt_id,
        }
        return egs

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        return self.extract_feature(self.file_list[index])

def test_datareader():
    file_name = '../wav_list/tt/linear_20.lst'
    sample_rate = 16000
    loader = DataReader(file_name, sample_rate)
    cnt = 0
    for egs in loader:
        print(cnt)
        print("inputs.shape: ", egs['mix'].shape)
        print("utt_id: ", egs['utt_id'])
        print()
        cnt = cnt + 1
        if cnt >= 10:
            break


if __name__ == "__main__":
    test_datareader()