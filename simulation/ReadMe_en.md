
# Simulation set synthesis

## 1. The simulation rirs
Parameter and path in configuration `tencent_challenge_rirgenerator.py` file
```bash 
python ./tencent_challenge_rirgenerator.py
```
If you use the supplied RIR, this step is not required.

## 2. Prepare speech and noise sets
### 2.1 Download the data
speech data:

[aishell-1](http://openslr.org/33/) 

[aishell-3](http://openslr.org/93/)

[librispeech-360](http://openslr.org/12/)

[VCTK](https://doi.org/10.7488/ds/2645)


noise data: 

[MUSAN](http://openslr.org/17/)

[Audioset](https://github.com/marc-moreaux/audioset_raw)

### 2.2 Generate the list file and partition the training and checksum sets

Configure the path to several datasets in './prepare. Sh '

```bash
bash ./prepare.sh
```


## 3. Generate the data

Configure using the supplied parameters
```
Attention to the data/[dev | "train"]_linear_mix. In the config file path is replaced with the corresponding path
```
```bash
# dev set of linear mic array 
    python mix_wav.py --mix_config_path=./data/dev_linear_simu_mix.config --save_dir=./data/wavs/dev/simu_linear/ --chunk_len=6 --generate_config=False 
```


Use the new parameter configuration

```bash 
# dev set of linear mic array 
    python mix_wav.py --clean_wav_list=./data/dev_clean.lst --noise_wav_list=./data/dev_noise.lst --rir_wav_list=./data/dev_linear_rir.lst --mix_config_path=./data/dev_linear_simu_mix.config --save_dir=./data/wavs/dev/simu_linear --chunk_len=6 --generate_config=True 
```
