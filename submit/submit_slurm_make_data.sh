#!/bin/bash

# Submission command:   sbatch submit_slurm_make_data.sh

#SBATCH --job-name=make_data_dijet_pt_160_eta_3.2_4.5
#SBATCH -p COMPUTE
#SBATCH --cpus-per-task=16

#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH -N1
#SBATCH -n4
# Your config has 34 event_fraction slices
#SBATCH --array=0-33

#SBATCH --mail-user=zcapcdo@ucl.ac.uk
#SBATCH --mail-type=ALL

#SBATCH --output=/home/xzcapcdo/lundtoptagger/logs/slurm-%j.%a.%x.out


# ------------------------------------------
# Environment and Paths
# ------------------------------------------
cd /home/xzcapcdo/lundtoptagger
echo "Now in $(pwd)"
hostname

echo "Activating environment..."
source /share/data1/xucaphue/setup.sh
conda activate /share/data1/xucaphue/envs/pytorch_py39_cu126
echo "Conda env: $CONDA_DEFAULT_ENV"


# ------------------------------------------
# Parameters
# ------------------------------------------
main_config="configs/config_make_data_SRJ.yaml"
signal_config="configs/config_signal_SRJ.yaml"

event_fraction_idx=${SLURM_ARRAY_TASK_ID}

echo "Using main config:   $main_config"
echo "Using signal config: $signal_config"
echo "event_fraction_idx:  $event_fraction_idx"


# ------------------------------------------
# Run MakeData
# ------------------------------------------
python Make_data_SRJ.py "$main_config" --override \
    signal_config_file="$signal_config" \
    id="dijet" \
    signal="srj" \
    event_fraction_idx="$event_fraction_idx"

echo "Job done."
