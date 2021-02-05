import os
import sys
import time

from pathlib import Path
from collections import defaultdict

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import os
from loss.loss import sisnr_loss

sys.path.append(
    os.path.dirname(__file__))
from logger.logger import get_logger

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

class SimpleTimer(object):
    '''
    A simple timer
    '''
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return (time.time() - self.start) / 60

class ProgressReporter(object):
    '''
    A sample progress reporter
    '''
    def __init__(self, logger, period=100):
        self.period = period
        if isinstance(logger, str):
            self.logger = get_logger(logger, file=True)
        else:
            self.logger = logger
        self.header = "Trainer"
        self.reset()
    
    def log(self, sstr):
        self.logger.info(f"{self.header}: {sstr}")
    
    def eval(self):
        self.log("set eval mode...")
        self.mode = "eval"
        self.reset()
    
    def train(self):
        self.log("set train mode...")
        self.mode = "train"
        self.reset()
    
    def reset(self):
        self.stats = defaultdict(list)
        self.timer = SimpleTimer()

    def add(self, key, value, batch_num, epoch):
        self.stats[key].append(value)
        N = len(self.stats[key])
        if not N % self.period:
            avg = sum(self.stats[key][-self.period:]) / self.period
            self.log(f"Epoch:{epoch} processed {N:.2e} / {batch_num:.2e} batches ({key} = {avg:+.2f})...")
        
    def report(self, epoch, lr):
        N = len(self.stats["loss"])
        if self.mode == "eval":
            sstr = ",".join(
                map(lambda f: "{:.2f}".format(f), self.stats["loss"]))
            self.log(f"loss on {N:d} batches: {sstr}")
        
        loss = sum(self.stats["loss"]) / N
        cost = self.timer.elapsed()
        sstr = f"Loss(time/N, lr={lr:.3e}) - Epoch {epoch:2d}: " + f"{self.mode} = {loss:.4f}({cost:.2f}m/{N:d})"
        return loss, sstr

class Trainer(object):
    '''
    Basic neural network trainer
    '''
    def __init__(self,
                 nnet,
                 optimizer,
                 scheduler,
                 device,
                 conf):
        
        self.default_device = device

        self.checkpoint = Path(conf['train']['checkpoint'])
        self.checkpoint.mkdir(exist_ok=True, parents=True)
        self.reporter = ProgressReporter(
            (self.checkpoint / "trainer.log").as_posix() if conf['logger']['path'] is None else conf['logger']['path'],
            period=conf['logger']['print_freq'])
        
        self.gradient_clip = conf['optim']['gradient_clip']
        self.start_epoch = 0 # zero based
        self.no_impr = conf['train']['early_stop']
        self.save_period = conf['train']['save_period']

        # only network part
        self.num_params = sum(
            [param.nelement() for param in nnet.parameters()]) / 10.0**6
        
        # logging
        self.reporter.log("model summary:\n{}".format(nnet))
        self.reporter.log(f"#param: {self.num_params:.2f}M")
        
        if conf['train']['resume']:
            # resume nnet and optimizer from checkpoint
            if not Path(conf['train']['resume']).exists():
                raise FileNotFoundError(
                    f"Could not find resume checkpoint: {conf['train']['resume']}")
            cpt = th.load(conf['train']['resume'], map_location="cpu")
            self.start_epoch = cpt["epoch"]
            self.reporter.log(
                f"resume from checkpoint {conf['train']['resume']}: epoch {self.start_epoch:d}")
            # load nnet
            nnet.load_state_dict(cpt["model_state_dict"])
            self.nnet = nnet.to(self.default_device)
            optimizer.load_state_dict(cpt["optim_state_dict"])
            self.optimizer = optimizer
        else:
            self.nnet = nnet.to(self.default_device)
            self.optimizer = optimizer
        
        if conf['optim']['gradient_clip']:
            self.reporter.log(
                f"gradient clipping by {conf['optim']['gradient_clip']}, default L2")
            self.clip_norm = conf['optim']['gradient_clip']
        else:
            self.clip_norm = 0
        
        self.scheduler = scheduler

    def save_checkpoint(self, epoch, best=True):
        '''
        Save checkpoint (epoch, model, optimizer)
        '''
        cpt = {
            "epoch": epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }
        cpt_name = "{0}.pt.tar".format("best" if best else "last")
        th.save(cpt, self.checkpoint / cpt_name)
        self.reporter.log(f"save checkpoint {cpt_name}")
        if self.save_period > 0 and epoch % self.save_checkpoint == 0:
            th.save(cpt, self.checkpoint / f"{epoch}.pt.tar")

    def train(self, data_loader, epoch):
        self.nnet.train()
        self.reporter.train()
        batch_num = len(data_loader)

        for egs in data_loader:
            # load to gpu
            egs = load_obj(egs, self.default_device)
            egs["mix"] = egs["mix"].contiguous()
            egs["ref"] = egs['ref'].contiguous()
            self.optimizer.zero_grad()
            est = nn.parallel.data_parallel(self.nnet, (egs["mix"]))

            loss = sisnr_loss(est["wav"], egs["ref"][:, 0])
            loss.backward()
            self.reporter.add("loss", loss.item(), batch_num, epoch)

            if self.gradient_clip:
                norm = clip_grad_norm_(self.nnet.parameters(),
                                       self.gradient_clip)
                self.reporter.add("norm", norm, batch_num, epoch)
            self.optimizer.step()
    
    def eval(self, data_loader, epoch):
        self.nnet.eval()
        self.reporter.eval()
        batch_num = len(data_loader)

        with th.no_grad():
            for egs in data_loader:
                egs = load_obj(egs, self.default_device)
                egs["mix"] = egs["mix"].contiguous()
                egs["ref"] = egs["ref"].contiguous()
                est = nn.parallel.data_parallel(self.nnet, (egs["mix"]))

                loss = sisnr_loss(est["wav"], egs["ref"][:, 0])
                self.reporter.add("loss", loss.item(), batch_num, epoch)
    
    def run(self, train_loader, valid_loader, num_epoches=50):
        '''
        Run on whole training set and evaluate
        '''
        # make dilated conv faster
        th.backends.cudnn.benchmark = True
        # avoid alloc memory grom gpu0
        # th.cuda.set_device(self.default_device)

        e = self.start_epoch
        self.eval(valid_loader, e)
        best_loss, _ = self.reporter.report(e, 0)
        self.reporter.log(f"start from epoch {e:d}, loss = {best_loss:.4f}")
        # make sure not inf
        self.scheduler.best = best_loss
        no_impr = 0

        while e < num_epoches:
            e += 1
            cur_lr = self.optimizer.param_groups[0]["lr"]

            # >> train
            self.train(train_loader, e)
            _, sstr = self.reporter.report(e, cur_lr)
            self.reporter.log(sstr)
            # << train
            # >> eval
            self.eval(valid_loader, e)
            cv_loss, sstr = self.reporter.report(e, cur_lr)
            if cv_loss > best_loss:
                no_impr += 1
                sstr += f"| no impr, best = {self.scheduler.best:.4f}"
            else:
                best_loss = cv_loss
                no_impr = 0
                self.save_checkpoint(e, best=True)
            self.reporter.log(sstr)
            # << eval
            # schedule here
            self.scheduler.step(cv_loss)
            # flush scheduler info
            sys.stdout.flush()
            # save checkpoint
            self.save_checkpoint(e, best=False)
            if no_impr == self.no_impr:
                self.reporter.log(
                    f"stop training cause no impr for {no_impr:d} epochs")
                break
        self.reporter.log(f"training for {e:d}/{num_epoches:d} epoches done!")


