CUDA_VISIBLE_DEVICES=3
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file /data/data/arrv/ThinkTuning_v1/deepspeed_zero3.yaml train.py 