#!/bin/bash
#SBATCH --job-name=fm_pretrain
#SBATCH --partition=suma_a6000,gigabyte_a6000,gigabyte_a5000,asus_a5000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/pretrain_%j.log
#SBATCH --error=logs/pretrain_%j.log

echo "=== Job started: $(date) ==="
echo "Node: $SLURMD_NODENAME"
nvidia-smi | head -15

cd /home/kyn7666/IngChicken-FM

/home/kyn7666/anaconda3/envs/lerobot/bin/python -m scripts.train_pretrain \
    --config configs/pretrain.yaml

echo "=== Job finished: $(date) ==="
