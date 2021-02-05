#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" train code, the experiment parameters are in ./config/train.yaml
@author: nwpuykjv@163.com
"""

import pprint
import argparse
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import torch as th
import numpy as np
from pathlib import Path

import sys
import os

from loader.fix_chunk_dataloader import make_fix_loader
from loader.config_dataloader import make_config_loader
from trainer.trainer import Trainer
from model.new_lstm_dnn_cirm import Nnet as model

def make_optimizer(params, opt):
    '''
    make optimizer
    '''
    supported_optimizer = {
        "sgd": th.optim.SGD,  # momentum, weight_decay, lr
        "rmsprop": th.optim.RMSprop,  # momentum, weight_decay, lr
        "adam": th.optim.Adam,  # weight_decay, lr
        "adadelta": th.optim.Adadelta,  # weight_decay, lr
        "adagrad": th.optim.Adagrad,  # lr, lr_decay, weight_decay
        "adamax": th.optim.Adamax  # lr, weight
        # ...
    }

    if opt['optim']['name'] not in supported_optimizer:
        raise ValueError("Now only support optimizer {}".format(opt['optim']['name']))
    optimizer = supported_optimizer[opt['optim']['name']](params, **opt['optim']['optimizer_kwargs'])
    return optimizer

def make_dataloader(opt):
    # make train's dataloader
    train_dataloader = make_config_loader(
        config_scp=opt['datasets']['train']['config_scp'],
        chunk=opt['datasets']['dataloader_setting']['train_chunk'],
        **opt['datasets']['dataloader_setting']['other'],
    )

    # make validation dataloader
    valid_dataloader = make_fix_loader(
        wav_scp=opt['datasets']['val']['wav_scp'],
        mix_dir=opt['datasets']['val']['mix_dir'],
        ref_dir=opt['datasets']['val']['ref_dir'],
        chunk=opt['datasets']['dataloader_setting']['val_chunk'],
        **opt['datasets']['dataloader_setting']['other'],
    )
    return train_dataloader, valid_dataloader

def run(args):

    print("Arguments in args:\n{}".format(pprint.pformat(vars(args))), flush=True)

    # load configurations
    with open(args.conf, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    print("Arguments in yaml:\n{}".format(pprint.pformat(conf)), flush=True)

    checkpoint_dir = Path(conf['train']['checkpoint'])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    random.seed(conf['train']['seed'])
    np.random.seed(conf['train']['seed'])
    th.cuda.manual_seed_all(conf['train']['seed'])

    # if exist, resume training
    last_checkpoint = checkpoint_dir / "last.pt.tar"
    if last_checkpoint.exists():
        print(f"Found old checkpoint: {last_checkpoint}", flush=True)
        conf['train']['resume'] = last_checkpoint.as_posix()

    # dump configurations
    with open(checkpoint_dir / "train.yaml", "w") as f:
        yaml.dump(conf, f)
    
    #build nnet
    nnet = model(**conf["nnet_conf"])
    # build optimizer
    optimizer = make_optimizer(nnet.parameters(), conf)
    # build dataloader
    train_loader, valid_loader = make_dataloader(conf)
    # build scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=conf['scheduler']['factor'],
        patience=conf['scheduler']['patience'],
        min_lr=conf['scheduler']['min_lr'],
        verbose=True)

    device = th.device('cuda' if conf['train']['use_cuda'] and th.cuda.is_available() else 'cpu')

    trainer = Trainer(nnet,
                      optimizer,
                      scheduler,
                      device,
                      conf)
    
    trainer.run(train_loader,
                valid_loader,
                num_epoches=conf['train']['epoch'],
                )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to train separation model in Pytorch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-conf",
                        type=str,
                        required=True,
                        help="Yaml configuration file for training")
    args = parser.parse_args()
    run(args)