import os
import sys
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as tf

sys.path.append(os.path.dirname(__file__))

from feature import IPDFeature, ConvSTFT, ConviSTFT

class Nnet(nn.Module):

    def __init__(self,
                 win_len=512,
                 win_inc=256,
                 fft_len=512,
                 win_type='hanning',
                 num_bins=257,
                 ipd_num=4,
                 rnn_units=512,
                 rnn_layers=3,
                 bidirectional=False,
                 ipd_index="0,4;1,5;2,6;3,7",
                 cos=True,
                 sin=False,
                ):
        super(Nnet, self).__init__()

        self.ipd_extractor = IPDFeature(ipd_index=ipd_index,
                                        cos=cos,
                                        sin=sin,
                                       )
        
        self.stft = ConvSTFT(win_len=win_len,
                             win_inc=win_inc,
                             fft_len=fft_len,
                             win_type=win_type,
                            )
        
        self.istft = ConviSTFT(win_len=win_len,
                               win_inc=win_inc,
                               fft_len=fft_len,
                               win_type=win_type,
                              )
        
        self.lstm = nn.LSTM(input_size=(2+ipd_num)*num_bins,
                            hidden_size=rnn_units,
                            num_layers=rnn_layers,
                            bidirectional=bidirectional,
                            batch_first=True,
                           )

        fac = 2 if bidirectional else 1
        self.linear = nn.Linear(rnn_units * fac,
                                num_bins * 2,
                                )

    def forward(self, x):
        # N x C x F x T
        _, p = self.stft(x, cplx=False)
        # N x MF x T
        ipd = self.ipd_extractor(p)
        # N x C x F x T
        r, i = self.stft(x, cplx=True)
        # N x F x T
        r_spec = r[:, 0]
        i_spec = i[:, 0]

        # N x (2 + M)F x T
        inp = th.cat([r_spec, i_spec, ipd], 1)
        # N x T x (2 + M)F
        inp = inp.permute(0, 2, 1)

        # N x T x H
        out, _ = self.lstm(inp)
        # N x T x 2F
        mask = self.linear(out)
        # N x T x F
        r_mask, i_mask = th.chunk(mask, 2, 2)
        # N x F x T
        r_mask = r_mask.permute(0, 2, 1)
        i_mask = i_mask.permute(0, 2, 1)

        # N x F x T
        r_out_spec = r_mask * r_spec - i_mask * i_spec
        i_out_spec = r_mask * i_spec + i_mask * r_spec
        # N x L
        out_wav = self.istft(r_out_spec, i_out_spec, cplx=True)
        out_wav = th.clamp(out_wav, -1, 1)

        est = {
            "wav": out_wav,
        }        
        return est

def test_nnet():
    net = Nnet()
    x = th.randn([3, 8, 16000 * 4])
    # net(x)
    est = net(x)
    print('est["wav"].shape: ', est["wav"].shape)

if __name__ == "__main__":
    test_nnet()

