CUDA_VISIBLE_DEVICES=0,1 python inference.py
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file /home/nkusumba/ThinkTuning_v1/deepspeed_zero3.yaml scripts/train.py 