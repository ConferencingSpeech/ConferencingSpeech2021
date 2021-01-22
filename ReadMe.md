
#  ConferencingSpeech 2021 chanllenge

This repository contains the datasets list and scripts required for the ConferencingSpeech challenge. For more details about the challenge, please see our [website](https://tea-lab.qq.com/conferencingspeech-2021/#/). 

# Details
- `baseline`, this folder contains baseline include inference model exported by onnx and inference scripts;
- `eval`, this folder contains evaluation scripts to calculate PESQ, STOI and SI-SNR;
- `selected_lists`, the selected wave about train speech and noise wave name from [aishell-1](http://openslr.org/33/), [aishell-3](http://openslr.org/93/), [librispeech-360](http://openslr.org/12/), [VCTK](https://doi.org/10.7488/ds/2645), [MUSAN](http://openslr.org/17/), [Audioset](https://github.com/marc-moreaux/audioset_raw)

- `simulation`, about simulation scripts, details to see [ReadMe](./simulation/ReadMe.md) 
- `requirements.txt`, dependency

# requirements
python3.6 or above

```python 
pip -r install requirements.txt
```



# code license 
[Apache 2.0](./license)
