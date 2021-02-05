import torch
import torch as th
import torch.nn as nn

def remove_dc(data):
    mean = torch.mean(data, -1, keepdim=True)
    data = data - mean
    return data

def si_snr(s1, s2, eps=1e-8):
    # s1 = remove_dc(s1)
    # s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target =  s1_s2_norm/(s2_s2_norm+eps)*s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
    return torch.mean(snr)


def l2_norm(s1, s2):
    #norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    #norm = torch.norm(s1*s2, 1, keepdim=True)
    
    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm 

def sisnr_loss(inputs, labels):
    return -(si_snr(inputs, labels))

def test():
    x = th.randn((9)).reshape(3, 3)
    y = th.randn((9)).reshape(3, 3)
    print(x)
    print(y)
    z = sisnr_loss(x, y)
    print(z)
    print(z.item())

if __name__ == "__main__":
    test()
