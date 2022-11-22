# CUDA_VISIBLE_DEVICES=0 python3 train.py --pretrained tae898/emoberta-large --dataset DACON --batch 4 --wandb;
CUDA_VISIBLE_DEVICES=1 python3 train.py --pretrained tae898/emoberta-large --dataset DACON --batch 4 --loss focal_loss --freeze --wandb;
CUDA_VISIBLE_DEVICES=0 python3 train.py --pretrained tae898/emoberta-large --dataset DACON --batch 4 --loss focal_loss --wandb;
# CUDA_VISIBLE_DEVICES=1 python3 train.py --pretrained roberta-large --dataset DACON --batch 4 --freeze --wandb;