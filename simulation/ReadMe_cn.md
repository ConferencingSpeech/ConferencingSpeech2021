
# 训练集与仿真集合成

## 1. 仿真rir
配置文件中参数及路径
```bash 
python ./tencent_challenge_rirgenerator.py
```
如果使用提供的RIR，则不需要这步 

## 2. 准备语音与噪声集合 
### 2.1 下载数据 
语音数据 

[aishell-1](http://openslr.org/33/) 

[aishell-3](http://openslr.org/93/)

[librispeech-360](http://openslr.org/12/)

[VCTK](https://doi.org/10.7488/ds/2645)


噪声数据 

[MUSAN](http://openslr.org/17/)

[Audioset](https://github.com/marc-moreaux/audioset_raw)

### 2.2 生成list并筛选

配置 `./prepare.sh` 中的几个数据集的路径

```bash
bash ./prepare.sh
```


## 3. 生成数据

使用提供的参数配置 
```
注意，要将 data/[dev|train]_linear_mix.config 中文件的路径替换成对应的路径
```
```bash
# dev set of linear mic array 
    python mix_wav.py --mix_config_path=./data/dev_linear_simu_mix.config --save_dir=./data/wavs/dev/simu_linear/ --chunk_len=6 --generate_config=False 
```


使用新的参数配置

```bash 
# dev set of linear mic array 
    python mix_wav.py --clean_wav_list=./data/dev_clean.lst --noise_wav_list=./data/dev_noise.lst --rir_wav_list=./data/dev_linear_rir.lst --mix_config_path=./data/dev_linear_simu_mix.config --save_dir=./data/wavs/dev/simu_linear --chunk_len=6 --generate_config=True 
```
