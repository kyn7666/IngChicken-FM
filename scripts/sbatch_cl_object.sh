#!/bin/bash
#SBATCH --job-name=cl_object
#SBATCH --partition=suma_a6000,gigabyte_a6000,gigabyte_a5000,asus_a5000
#SBATCH --gres=gpu:1
#SBATCH --exclude=node25
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/cl_object_%j.log
#SBATCH --error=logs/cl_object_%j.log

echo "=== Job started: $(date) ==="
echo "Node: $SLURMD_NODENAME"

cd /home/kyn7666/IngChicken-FM

/home/kyn7666/anaconda3/envs/lerobot/bin/python -m scripts.train_sequential \
    --config configs/cl_object.yaml \
    --pretrain-ckpt checkpoints/pretrain_fm_1000ep/best_ema.pt

echo "=== Job finished: $(date) ==="
