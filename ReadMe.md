
#  ConferencingSpeech 2021 challenge

This repository contains the datasets list and scripts required for the ConferencingSpeech challenge. For more details about the challenge, please see our [website](https://tea-lab.qq.com/conferencingspeech-2021/#/). 

# Details
- `baseline`, this folder contains baseline system include inference model exported by onnx and inference scripts;
- `eval`, this folder contains evaluation scripts to calculate PESQ, STOI and SI-SNR;
- `selected_lists`, the selected wave about train speech and noise wave name from [aishell-1](http://openslr.org/33/), [aishell-3](http://openslr.org/93/), [librispeech-360](http://openslr.org/12/), [VCTK](https://datashare.ed.ac.uk/handle/10283/2651), [MUSAN](http://openslr.org/17/), [Audioset](https://github.com/marc-moreaux/audioset_raw). **Each participant** is **only allowed** to use the selected **speech** and **noise** data below :
    - `selected_lists/dev/circle.name` circle RIR wave utt name of dev set 
    - `selected_lists/dev/linear.name` linear RIR wave utt name of dev set
    - `selected_lists/dev/non_uniform.name` non uniform linear RIR wave utt name of dev set
    - `selected_lists/dev/clean.name` wave utt name of dev set used clean set
    - `selected_lists/dev/noise.name` wave utt name of dev set used noise set
    - `selected_lists/train/aishell_1.name` wave utt name from aishell-1 set used in train set
    - `selected_lists/train/aishell_3.name` wave utt name from aishell-3 set used in train set
    - `selected_lists/train/librispeech_360.name` wave utt name from librispeech-360 set used in train set
    - `selected_lists/train/vctk.name` wave utt name from VCTK set used in train set
    - `selected_lists/train/audioset.name` wave utt name from Audioset used in train set
    - `selected_lists/train/musan.name` wave utt name from MUSAN used in train set
    - `selected_lists/train/circle.name` circle wave RIR name of train set 
    - `selected_lists/train/linear.name` linear wave RIR name of train set
    - `selected_lists/train/non_uniform.name` non unifrom linear RIR utt name of train set
- `simulation`, about simulation scripts, how to use to see [ReadMe](./simulation/ReadMe.md) 
    - `simulation/mix_wav.py` simulate dev set and train set
    - `simulation/prepare.sh` use `selected_lists/*/*name` to select used wave from downloaded raw data, or you can select them by yourself scripts.
    - `simulation/quick_select.py` quickly select the name by a name list instead of `grep -r -f`
    - `simulation/challenge_rirgenerator.py` the script to simulate RIRs in train and dev set
    - `simulation/data/dev_circle_simu_mix.config` dev circle set simulation setup, include clean wave, noise wave, rir wave, snr, volume scale, start point
    - `simulation/data/dev_linear_simu_mix.config` dev linear set simulation setup, include clean wave, noise wave, rir wave, snr, volume scale, start point
    - `simulation/data/dev_non_uniform_linear_simu_mix.config` dev non uniform linear set simulation setup, include clean wave, noise wave, rir wave, snr, volume scale, start point
    - `simulation/data/train_simu_circle.config` train circle set simulation setup, include clean wave, noise wave, rir wave, snr, volume scale, start point; please download it from dropbox.
    - `simulation/data/train_simu_linear.config` train linear set simulation setup, include clean wave, noise wave, rir wave, snr, volume scale, start point; please download it from dropbox.
    - `simulation/data/train_simu_non_uniform.config` train non uniform linear set simulation setup, include clean wave, noise wave, rir wave, snr, volume scale, start point; please download it from dropbox.

- `requirements.txt`, dependency

**Notes:** 

    1. \*.config file should be replaced with correct path of audio files.
    2. Training config files have been released together with challenge data.

# Citation
If you use this challenge dataset and baseline system in a publication, please cite the following paper:

@article{wei2021interspeech,
  title={{INTERSPEECH 2021 ConferencingSpeech Challenge: Towards Far-field Multi-Channel Speech Enhancement for Video Conferencing}},
  author={Wei Rao and Yihui Fu and Yanxin Hu and Xin Xu and Yvkai Jv and Jiangyu Han and Zhongjie Jiang and Lei Xie and Yannan Wang and Shinji Watanabe and Zheng-Hua Tan and Hui Bu and Tao Yu and Shidong Shang},
  journal={arXiv preprint arXiv:2104.00960}
}

# Requirements
python3.6 or above

```python 
pip install -r requirements.txt
```
if you simulation RIRs by yourself with our scripts, you may better install this:

[pyrirgen](https://github.com/Marvin182/rir-generator/tree/master/python)

# Code license 

[Apache 2.0](./LICENSE)
