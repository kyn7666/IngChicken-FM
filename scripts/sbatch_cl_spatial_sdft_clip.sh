#!/bin/bash
#SBATCH --job-name=cl_spatial_sdft_clip
#SBATCH --partition=suma_a6000,gigabyte_a6000,gigabyte_a5000,asus_a5000
#SBATCH --gres=gpu:1
#SBATCH --exclude=node25,node28
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/cl_spatial_sdft_clip_%j.log
#SBATCH --error=logs/cl_spatial_sdft_clip_%j.log

echo "=== Job started: $(date) ==="
echo "Node: $SLURMD_NODENAME"

cd /home/kyn7666/IngChicken-FM

/home/kyn7666/anaconda3/envs/lerobot/bin/python -m scripts.train_sequential_sdft \
    --config configs/cl_spatial_sdft_clip_l0.1_s220.yaml

echo "=== Job finished: $(date) ==="
