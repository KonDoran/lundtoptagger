#!/bin/bash
#SBATCH -p GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=12:00:00
#SBATCH --job-name=lundtoptagger_eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=zcapcdo@ucl.ac.uk      # TODO: change this
#SBATCH --mail-type=ALL

set -euo pipefail

echo "===== [1/7] Job started (EVAL) ====="
date
echo "SLURM job name:   $SLURM_JOB_NAME"
echo "SLURM job id:     $SLURM_JOB_ID"
echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "Node:             ${SLURMD_NODENAME:-UNKNOWN}"
echo "============================================="

# Work from the directory where the script + Python files live
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "===== [2/7] Sourcing environment ====="
# Same environment you used for training
source /share/data1/xucaphue/setup.sh
echo "Environment sourced."
echo "============================================="

echo "===== [3/7] Thread configuration ====="
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "============================================="

echo "===== [4/7] GPU sanity check ====="
hostname
nvidia-smi || echo "WARNING: nvidia-smi failed (no GPU visible?)"
echo "============================================="

echo "===== [5/7] Python / torch sanity check ====="
which python || echo "No python in PATH!"
python --version || echo "python --version failed"
python - << 'PY'
import sys
print("sys.executable:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("ERROR importing torch:", e)
PY
echo "============================================="

echo "===== [6/7] Setting up checkpoints & branch names ====="

# TODO: update these to YOUR checkpoint paths & desired branch names
checkpoints=(
    "/home/xzcapcdo/lundtoptagger/Models/LundNetBase/QCD_part_0_0.25_ln_kT_cut_None_with_pt/LundNet_part0_0.25percent_ln_kT_cut_None_with_pt_e035_0.09073.pt"
)

scores_branch_names=(
    fjet_{model}_score_loss_0.09073
)

# Sanity check: arrays must be same length
if [ "${#checkpoints[@]}" -ne "${#scores_branch_names[@]}" ]; then
    echo "ERROR: checkpoints and scores_branch_names arrays have different lengths!"
    echo "checkpoints:        ${#checkpoints[@]}"
    echo "scores_branch_names:${#scores_branch_names[@]}"
    exit 1
fi

echo "Number of checkpoints to process: ${#checkpoints[@]}"
echo "============================================="

echo "===== [7/7] Starting evaluation loop ====="

for i in "${!checkpoints[@]}"; do
    ckpt="${checkpoints[$i]}"
    branch="${scores_branch_names[$i]}"

    echo "---------------------------------------------"
    echo "[EVAL] Index: $i"
    echo "[EVAL] Checkpoint: $ckpt"
    echo "[EVAL] Scores branch name: $branch"

    if [ ! -f "$ckpt" ]; then
        echo "WARNING: checkpoint file does NOT exist: $ckpt"
        echo "Skipping..."
        continue
    fi

    start_time=$(date +%s)

    python test_make_scores.py configs/config_make_scores.yaml  #--override \
    #    data.sample="v2.2.0_GN2X/data_part5_2.00percent" \
    #   data.paths_to_test_file_root="[ \
    #        '/home/tmlinare/Lund_tagging/lundtoptagger_graphs_lustre/{sample}/*W_flat_pt*.root', \
    #        '/home/tmlinare/Lund_tagging/lundtoptagger_graphs_lustre/{sample}/*QCD*.root' \
    #    ]" \
    #    data.paths_to_test_file_graphs="[ \
    #        '/home/tmlinare/Lund_tagging/lundtoptagger_graphs_lustre/{sample}/graphs*W_flat_pt*', \
    #        '/home/tmlinare/Lund_tagging/lundtoptagger_graphs_lustre/{sample}/graphs*QCD*' \
    #    ]" \
    #    data.path_to_outdir="/home/tmlinare/Lund_tagging/lundtoptagger_scores_lustre/scores_v2.2.6_data_{sample}" \
    #    data.output_suffix="_scores" \
    #    test.path_to_combined_ckpt.null="$ckpt" \
    #    test.scores_branch_name="$branch"

    status=$?
    end_time=$(date +%s)
    dt=$(( (end_time - start_time) / 60 ))

    if [ $status -eq 0 ]; then
        echo "[EVAL] Finished checkpoint index $i successfully in ${dt} minutes."
    else
        echo "[EVAL] ERROR: checkpoint index $i failed with exit code $status (after ${dt} minutes)."
    fi
    echo "---------------------------------------------"
    echo ""
done

echo "All checkpoints processed (or skipped)."
date
echo "===== EVAL job completed ====="
