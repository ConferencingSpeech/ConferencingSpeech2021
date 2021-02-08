# Details
Preparation files before training and inference
    1. Generate train configuration file like "./wav_list/tr/linear_100.lst" (Because the full 
        amount of data occupies too much space, so we mix them dynamicly when training)
        and "config_scp" parameter in the "./config/train.yaml"
            The format of the configuration file is as follows:
                clean_path start_time noise_path rir_path snr scale
    2. Generate dev wav list like "./wav_list/cv/linear_20.lst" and then put the dev data in 
        the corresponding directory ./dev/xxx/{mix, ref}. After that, you should modify the 
        "wav_scp", "mix_dir", "ref_dir" parameters in the "./config/train.yaml" 
    3. If you want to run the inference code, please generate inference 
        wav list like "./wav_list/tt/linear_20.lst" and then modify the 
        "file_name" parameter in the "./config/test.yaml"

Training this model:
    1: If you want to adjust the network parameters and the path of 
    the training file, please modify the "./config/train.yaml" file.
    2: Training Command
        python train.py -conf ./config/train.yaml

Inference this model:
    1: If you want to adjust the path of the inferencing file, please 
    modify the "./config/test.yaml" file.
    2: Inference Command
        python test.py -conf ./config/test.yaml

In the train yaml file:
    conf['train']['checkpoint'] means the path to save experimental model

In the test yaml file:
    conf['test']['checkpoint'] means the path to load experimental model
    conf['save']['dir'] means the path to save generated wav

# Code license 

[Apache 2.0](../LICENSE)
