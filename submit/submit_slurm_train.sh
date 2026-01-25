#!/bin/bash

# ------------------------------------------------------------
#  SLURM config
# ------------------------------------------------------------

#SBATCH --job-name=SRJ_LundNet_train_5.00percent  # Job name
#SBATCH --time=24:00:00            # Time limit hrs:min:sec
#SBATCH -p GPU                     # GPU partition
#SBATCH -N1                        # single node
#SBATCH -n8                        # 8 CPU cores
#SBATCH --gres=gpu:1               # request 1 GPU
#SBATCH --mem=50G                  # 50 GB RAM

# Email notification (optional)
#SBATCH --mail-user=zcapcdo@ucl.ac.uk
#SBATCH --mail-type=ALL

# Log files
#SBATCH --output=/home/xzcapcdo/lundtoptagger/logs/slurm-%j.%x.out

# Submit by sbatch submit_slurm_train.sh

# ------------------------------------------------------------
#  Environment setup
# ------------------------------------------------------------

echo "===> Current directory before cd:"
pwd

# Move into SRJ project folder
cd /home/xzcapcdo/lundtoptagger/
echo "===> Switched directory to:"
pwd

echo "===> Hostname:"
hostname

echo "===> Activating conda environment..."
source /share/data1/xucaphue/setup.sh
conda activate /share/data1/xucaphue/envs/pytorch_py39_cu126
echo "Activated env: $CONDA_DEFAULT_ENV"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ------------------------------------------------------------
#  Run training
# ------------------------------------------------------------

echo ""
echo "===> Starting SRJ training..."
python weight_ONLY_TRAINS_SRJ.py configs/config_ONLY_TRAIN_SRJ.yaml

echo "===> Job finished."
