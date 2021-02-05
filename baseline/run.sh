# train
CUDA_VISIBLE_DEVICES=0 python train.py -conf ./config/train.yaml

# inference
CUDA_VISIBLE_DEVICES=0 python test.py -conf ./config/test.yaml

