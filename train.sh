CUDA_VISIBLE_DEVICES=0,1,2,3 
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file /data/data/arrv/ThinkTuning_v1/deepspeed_zero3.yaml scripts/train.py 