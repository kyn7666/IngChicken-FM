#!/bin/bash
#SBATCH --job-name=drift_ext
#SBATCH --partition=suma_a6000,gigabyte_a6000,gigabyte_a5000,asus_a5000
#SBATCH --gres=gpu:1
#SBATCH --exclude=node25,node28
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/drift_extended_%j.log
#SBATCH --error=logs/drift_extended_%j.log

echo "=== Job started: $(date) ==="
echo "Node: $SLURMD_NODENAME"

cd /home/kyn7666/IngChicken-FM

/home/kyn7666/anaconda3/envs/lerobot/bin/python -u -m scripts.visualize_drift_extended

echo "=== Done: $(date) ==="
