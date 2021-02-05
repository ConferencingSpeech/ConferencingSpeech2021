import math
import torch as th
import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.signal import get_window
import numpy as np

EPSILON = th.finfo(th.float32).eps
MATH_PI = math.pi

def init_kernels(win_len,
                 win_inc,
                 fft_len,
                 win_type=None,
                 invers=False):
    if win_type == 'None' or win_type is None:
        # N 
        window = np.ones(win_len)
    else:
        # N
        window = get_window(win_type, win_len, fftbins=True)#**0.5
    N = fft_len
    # N x F
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    # N x F
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    # 2F x N
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T
    if invers :
        kernel = np.linalg.pinv(kernel).T 

    # 2F x N * N => 2F x N
    kernel = kernel*window
    # 2F x 1 x N
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None,:,None].astype(np.float32))


class ConvSTFT(nn.Module):

    def __init__(self, 
                 win_len,
                 win_inc,
                 fft_len=None,
                 win_type='hamming',
                #  fix=True
                 ):
        super(ConvSTFT, self).__init__() 
        
        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        
        # 2F x 1 x N
        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
        #self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs, cplx=False):
        if inputs.dim() == 2:
            # N x 1 x L
            inputs = torch.unsqueeze(inputs, 1)
            inputs = F.pad(inputs,[self.win_len-self.stride, self.win_len-self.stride])
            # N x 2F x T
            outputs = F.conv1d(inputs, self.weight, stride=self.stride)
            # N x F x T
            r, i = th.chunk(outputs, 2, dim=1)
        else:
            N, C, L = inputs.shape
            inputs = inputs.view(N * C, 1, L)
            # NC x 1 x L
            inputs = F.pad(inputs, [self.win_len-self.stride, self.win_len-self.stride])
            # NC x 2F x T
            outputs = F.conv1d(inputs, self.weight, stride=self.stride)
            # N x C x 2F x T
            outputs = outputs.view(N, C, -1, outputs.shape[-1])
            # N x C x F x T
            r, i = th.chunk(outputs, 2, dim=2)
        if cplx:
            return r, i
        else:
            mags = th.clamp(r**2 + i**2, EPSILON)**0.5
            phase = th.atan2(i+EPSILON, r+EPSILON)
            return mags, phase

class ConviSTFT(nn.Module):

    def __init__(self, 
                 win_len, 
                 win_inc, 
                 fft_len=None, 
                 win_type='hamming', 
                #  fix=True
                 ):
        super(ConviSTFT, self).__init__() 
        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        
        # kernel: 2F x 1 x N
        # window: 1 x N x 1
        kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_type, invers=True)
        #self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.win_type = win_type
        self.win_len = win_len
        self.stride = win_inc
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer('window', window)
        self.register_buffer('enframe', torch.eye(win_len)[:,None,:])

    def forward(self, inputs, phase, cplx=False):
        """
        inputs : [B, N//2+1, T] (mags, real)
        phase: [B, N//2+1, T] (phase, imag)
        """ 

        if cplx:
            # N x 2F x T
            cspec = torch.cat([inputs, phase], dim=1)
        else:
            # N x F x T
            real = inputs*torch.cos(phase)
            imag = inputs*torch.sin(phase)
            # N x 2F x T
            cspec = torch.cat([real, imag], dim=1)
        # N x 1 x L
        outputs = F.conv_transpose1d(cspec, self.weight, stride=self.stride)

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        # 1 x N x T
        t = self.window.repeat(1,1,inputs.size(-1))**2
        # 1 x 1 x L
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        outputs = outputs/(coff+1e-8)
        #outputs = torch.where(coff == 0, outputs, outputs/coff)
        # N x 1 x L
        outputs = outputs[...,self.win_len-self.stride:-(self.win_len-self.stride)]
        # N x L
        outputs = outputs.squeeze(1)
        return outputs


class IPDFeature(nn.Module):
    """
    Compute inter-channel phase difference
    """
    def __init__(self,
                 ipd_index="1,0;2,0;3,0;4,0;5,0;6,0",
                 cos=True,
                 sin=False):
        super(IPDFeature, self).__init__()
        split_index = lambda sstr: [
            tuple(map(int, p.split(","))) for p in sstr.split(";")
        ]
        # ipd index
        pair = split_index(ipd_index)
        self.index_l = [t[0] for t in pair]
        self.index_r = [t[1] for t in pair]
        self.ipd_index = ipd_index
        self.cos = cos
        self.sin = sin
        self.num_pairs = len(pair) * 2 if cos and sin else len(pair)

    def extra_repr(self):
        return f"ipd_index={self.ipd_index}, cos={self.cos}, sin={self.sin}"

    def forward(self, p):
        """
        Accept multi-channel phase and output inter-channel phase difference
        args
            p: phase matrix, N x C x F x T
        return
            ipd: N x MF x T
        """
        if p.dim() not in [3, 4]:
            raise RuntimeError(
                "{} expect 3/4D tensor, but got {:d} instead".format(
                    self.__name__, p.dim()))
        # C x F x T => 1 x C x F x T
        if p.dim() == 3:
            p = p.unsqueeze(0)
        N, _, _, T = p.shape
        pha_dif = p[:, self.index_l] - p[:, self.index_r]
        if self.cos:
            # N x M x F x T
            ipd = th.cos(pha_dif)
            if self.sin:
                # N x M x 2F x T
                ipd = th.cat([ipd, th.sin(pha_dif)], 2)
        else:
            ipd = th.fmod(pha_dif, 2 * math.pi) - math.pi
        # N x MF x T
        ipd = ipd.view(N, -1, T)
        # N x MF x T
        return ipd

def test_stft():
    stft = ConvSTFT(512, 256, 512)
    
    single = True
    # single = False

    if single:
        x = th.randn([3, 16000 * 4])
        mag, phase = stft(x, cplx=False)
        # print('mag.shape: ', mag.shape)

    else:
        x = th.randn([3, 8, 16000 * 4])
        mag, phase = stft(x)
        # print('mag.shape: ', mag.shape)

    istft = ConviSTFT(512, 256, 512)
    wav = istft(mag, phase, cplx=False)
    print('wav.shape: ', wav.shape)


if __name__ == "__main__":
    test_stft()