#!/bin/bash
#SBATCH --job-name=srj_preprocess_160_eta_3.2_4.5_5percent
#SBATCH -p COMPUTE       # CPU Partition
# exclude nodes that do NOT mount /share/lustre properly
#SBATCH -N 1
#SBATCH -n 16                   # 16 CPU cores
#SBATCH --mem=240G              # 240GB Memory (KDE can use)
#SBATCH --time=72:00:00         # Up to three days
#SBATCH --export=ALL
#SBATCH --output=/home/xzcapcdo/lundtoptagger/preprocess_log/pt_160_eta_3.2_4.5/slurm-%j.out
#SBATCH --error=/home/xzcapcdo/lundtoptagger/preprocess_log/pt_160_eta_3.2_4.5/slurm-%j.err
#SBATCH --mail-user=zcapcdo@ucl.ac.uk
#SBATCH --mail-type=ALL
# submit by sbatch submit_preprocess.sh

echo "===> Hostname:"
hostname

echo "===> Activating environment"
source /share/data1/xucaphue/setup.sh
conda activate /share/data1/xucaphue/envs/pytorch_py39_cu126
echo "Using environment: $CONDA_DEFAULT_ENV"

echo "===> Running CPU preprocess script..."
cd /home/xzcapcdo/lundtoptagger

python preprocess_SRJ_CPU.py configs/config_preprocess_SRJ.yaml

echo "===> Job finished."
