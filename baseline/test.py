#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" inference code, the experiment parameters are in ./config/test.yaml;
@author: nwpuykjv@163.com
"""

import yaml
import torch as th
import torch.nn as nn
import numpy as np
from pathlib import Path
import soundfile as sf
import argparse
from tqdm import tqdm

from loader.datareader import DataReader
from model.new_lstm_dnn_cirm import Nnet as model

import sys
import os

def load_obj(obj, device):
    '''
    Offload tensor object in obj to cuda device
    '''
    def cuda(obj):
        return obj.to(device) if isinstance(obj, th.Tensor) else obj
    
    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)

def run(args):
    with open(args.conf, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    data_reader = DataReader(**conf['datareader'])
    device = th.device('cuda' if conf['test']['use_cuda'] and th.cuda.is_available() else 'cpu')

    nnet = model(**conf["nnet_conf"])

    checkpoint_dir = Path(conf['test']['checkpoint'])
    cpt_fname = checkpoint_dir / "best.pt.tar"
    cpt = th.load(cpt_fname, map_location="cpu")
    nnet.load_state_dict(cpt["model_state_dict"])
    nnet = nnet.to(device)
    nnet.eval()

    with th.no_grad():
        for egs in tqdm(data_reader):
            egs = load_obj(egs, device)
            egs['mix'] = egs['mix'].contiguous()
            est = nn.parallel.data_parallel(nnet, (egs["mix"]))
            out = est["wav"].detach().squeeze().cpu().numpy()
            sf.write(os.path.join(conf['save']['dir'], egs['utt_id'] + '.wav'), out, conf['save']['sample_rate'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to test separation model in Pytorch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-conf",
                        type=str,
                        required=True,
                        help="Yaml configuration file for training")
    args = parser.parse_args()
    run(args)


