#!/bin/bash

# to submit this script, do sbatch submit_slurm_scores.sh

# job name
# #SBATCH --job-name=lundtoptagger_full_a100_8CPUs
#SBATCH --job-name=lundtoptagger_eval

# choose the GPU queue
#SBATCH -p GPU

# request one node
#SBATCH -N1
# do not share nodes with other running jobs
# #SBATCH --exclusive
# exclude the node with 40 GB A100 GPUs - this ensures that if A100 GPUs are requested, the ones with 80 GB are used
# #SBATCH --exclude=compute-gpu-0-3

# keep environment variables
#SBATCH --export=ALL

# request CPUs
#SBATCH -n8

# request GPUs
#SBATCH --gres=gpu:1
# #SBATCH --constraint='a100|l40s|v100'
#SBATCH --constraint='v100'

# request enough memory
#SBATCH --mem=50G

# email notifications
#SBATCH --mail-user=toni.mlinarevic.20@ucl.ac.uk
#SBATCH --mail-type=ALL

# change log names; %j gives job id, %x gives job name
#SBATCH --output=/home/tmlinare/Lund_tagging/lundtoptagger_job_outputs/slurm-%j.%x.out
# optional separate error output file
# #SBATCH --error=/home/tmlinare/Lund_tagging/lundtoptagger_job_outputs/slurm-%j.%x.err

# speedup trick
# export OMP_NUM_THREADS=1

# TODO: loop over the checkpoints in the Python script instead so that data doesn't need to be repeatedly loaded; this takes up a significant amount of the processing time
# or submit a separate job for each checkpoint to run them all in parallel
checkpoints=(
    # "/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/models/hypatia_run12_v2.1.2_with_fix_adversarial_1-50_QCD_90percent_ln_kT_cut_None_new_pt_weights_2025-03-07_v100/LundNet_R22_ExtraNode_ln_kT_Cut_None_LRJ_NewData_Primary_comb_e063_0.17167.pt"
    # "/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/models/hypatia_run12_v2.1.2_with_fix_adversarial_1-50_QCD_90percent_ln_kT_cut_None_new_pt_weights_2025-03-07_v100/LundNet_R22_ExtraNode_ln_kT_Cut_None_LRJ_NewData_Primary_comb_e100_0.19217.pt"
    # "/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/models/hypatia_run12_v2.1.2_with_fix_adversarial_1-50_QCD_90percent_ln_kT_cut_None_new_pt_weights_2025-03-07_v100/LundNet_R22_ExtraNode_ln_kT_Cut_None_LRJ_NewData_Primary_comb_e200_0.64437.pt"
    # "/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/models/hypatia_run5_v2.0.2_adversarial_1-50_QCD_ln_kT_cut_None_new_pt_weights_2025-03-07_v100/LundNet_R22_ExtraNode_ln_kT_Cut_None_LRJ_NewData_Primarye200_0.06079_comb_.pt"
    "/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/models/hypatia_run17_v2.1.3_unfinished_Wwidemass_1-50_QCD_90percent_ln_kT_cut_None_new_pt_weights_2025-03-07_v100/LundNet_R22_ExtraNode_ln_kT_Cut_None_LRJ_NewData_Primary_e026_0.17686.pt"
    "/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/models/rvinasc_best_models/LundNet_R22_None_ln_kT_Cut_LRJ_Wtagging_Wflat_bt4800_Moredatae019_0.06902.pt"
    "/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/models/rvinasc_best_models/LundNet_R22_None_ln_kT_Cut_LRJ_Wtagging_bt4096_gss14_lossP5e320_0.04524_comb_.pt"
    # "/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/models/rvinasc_best_models/LundNet_R22_None_ln_kT_Cut_LRJ_Wtagging_bt4096_gss14_lossP5e198_0.04637_comb_.pt"
    # "/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/models/hypatia_run33_v2.1.5_adversarial_rafael_graphs/LundNet_R22_ExtraNode_ln_kT_Cut_None_LRJ_NewData_Primary_comb_e200_1.33664.pt"
    # "/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/models/hypatia_run34_v2.1.5_adversarial_rafael_graphs_SmallerW_Weights/LundNet_R22_ExtraNode_ln_kT_Cut_None_LRJ_NewData_Primary_comb_e200_0.03981.pt"
    # "/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/models/hypatia_run31_v2.1.5_top_1percent_flat_pt_masscut80/LundNet_R22_ExtraNode_ln_kT_Cut_None_LRJ_NewData_Primary_e031_0.05684.pt"
    # "/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/models/hypatia_run32_v2.1.5_top_1percent_flat_pt_masscut80_GN2X/LundNet_GN2X_R22_ExtraNode_ln_kT_Cut_None_LRJ_NewData_Primary_e021_0.03471.pt"
    "/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/models/jecifuen_checkpoints/2025_Models_1qcd_10W/LundNet_R22_No_ln_kT_Cut_LRJ_NewDatae017_0.62835.pt"
    "/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/models/hypatia_run4_v2.0.2_Wrealmass_1-50_QCD_ln_kT_cut_None_new_pt_weights_2025-03-07_v100/LundNet_R22_ExtraNode_ln_kT_Cut_None_LRJ_NewData_Primary_e018_0.01689.pt"
)
scores_branch_names=(
    # fjet_{model}_ann_score_e63
    # fjet_{model}_ann_score_e100
    # fjet_{model}_ann_score_e200
    # fjet_{model}_ann_score_run5_e200
    fjet_{model}_flatmass_run17  # used to call it just fjet_LundNet_score
    fjet_{model}_rafael_Wflat
    fjet_{model}_rafael_adv_e320
    # fjet_{model}_rafael_adv_e198
    # fjet_{model}_run33_rafael_graphs_adv_e200
    # fjet_{model}_run34_rafael_graphs_adv_e200
    fjet_{model}_jp_nominal
    fjet_{model}_nominal
)

cd ~/Lund_tagging/lundtoptagger
echo "Moved dir, now in:"
pwd

echo "Hostname:"
hostname

echo "Activating environment"
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate /share/rcifdata/tmlinare/conda/envs/pytorch_py39_cu126
echo $CONDA_DEFAULT_ENV

echo "CUDA_VISIBLE_DEVICES:"
echo $CUDA_VISIBLE_DEVICES

echo "Running training script..."
echo ""

for i in "${!checkpoints[@]}"; do
    ckpt="${checkpoints[$i]}"
    branch="${scores_branch_names[$i]}"
    echo "Processing checkpoint: $ckpt"
    echo "Using scores branch name: $branch"
    python test_make_scores.py configs/config_make_scores.yaml  --override \
        data.sample="v2.1.5_GN2X_m40-inf_pt200-3100_0.25percent" \
        data.paths_to_test_file_root="[ \
            '/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/graphs/{sample}/*W_flat_pt*.root', \
            '/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/graphs/{sample}/*QCD*.root' \
        ]" \
        data.paths_to_test_file_graphs="[ \
            '/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/graphs/{sample}/graphs*W_flat_pt*', \
            '/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/graphs/{sample}/graphs*QCD*' \
        ]" \
        data.path_to_outdir="/home/tmlinare/Lund_tagging/lundtoptagger_data_rcif/scores/data_{sample}_scores_v2.2.5/" \
        data.output_suffix="_scores" \
        test.path_to_combined_ckpt.null="$ckpt" \
        test.scores_branch_name="$branch"
done