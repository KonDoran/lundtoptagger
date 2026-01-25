#!/bin/bash
#SBATCH -p RCIF
#SBATCH --job-name=make_histos
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-18%10
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=zcapcdo@ucl.ac.uk
#SBATCH --mail-type=ALL

set -euo pipefail

echo "===== [1/8] Job start ====="
date
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID  ArrayIdx: ${SLURM_ARRAY_TASK_ID:-none}"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "Node: ${SLURMD_NODENAME:-UNKNOWN}"
mkdir -p logs
echo "==========================="

# Work from the directory where the script + python live
cd "$SLURM_SUBMIT_DIR"
echo "CWD: $(pwd)"

echo "===== [2/8] Sourcing environment ====="
# Use the shared DIAS environment (as in your other scripts)
source /share/data1/xucaphue/setup.sh
echo "Env sourced."
echo "Python: $(python -V 2>&1)"
echo "which python: $(which python)"
echo "============================="

# Threading hints for NumPy/MKL/ROOT
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
echo "Threads -> OMP:$OMP_NUM_THREADS MKL:$MKL_NUM_THREADS"

# ------------------ your arrays (unchanged) ------------------
JETM2_W_flatmass=(
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/JETM2_old/mc20_13TeV.802017.Py8EG_A14NNPDF23LO_WprimeWZ_flatpT_wideWmass.deriv.DAOD_JETM2.e8482_s3797_r13145_p5548_files_1-50.root \
     /share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/JETM2_old/mc20_13TeV.802017.Py8EG_A14NNPDF23LO_WprimeWZ_flatpT_wideWmass.deriv.DAOD_JETM2.e8482_s3797_r13145_p5548_files_51-99.root"
)

infile_lists=(
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.426345.e6880_s3681_r13144_p5981.FTAG1_TV3_ANALYSIS.root/*.root"
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.364703.e7142_s3681_r13144_p6453.FTAG1_TV3_ANALYSIS.root/*.root"
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.364704.e7142_s3681_r13144_p6453.FTAG1_TV3_ANALYSIS.root/*.root"
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.364705.e7142_s3681_r13144_p6453.FTAG1_TV3_ANALYSIS.root/*.root"
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.364706.e7142_s3681_r13144_p6453.FTAG1_TV3_ANALYSIS.root/*.root"
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.364707.e7142_s3681_r13144_p6453.FTAG1_TV3_ANALYSIS.root/*.root"
    "/share/lustreslow/tmlinare/Lund_tagging/jetmdatamc_output/JETM2_mc20_alternative_MC_Rafael/Herwing_dipole/*.root"
    "/share/lustreslow/tmlinare/Lund_tagging/jetmdatamc_output/JETM2_mc20_alternative_MC_Rafael/Sherpa_Cluster/*.root"
    "/share/lustreslow/tmlinare/Lund_tagging/jetmdatamc_output/JETM2_mc20_alternative_MC_Rafael/Sherpa_Lund/*.root"
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.801859.e8482_s3681_r13144_p6781.FTAG1_TV3_ANALYSIS.root/*.root"
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.364703.e7142_s3681_r13144_p6453.FTAG1_TV3_ANALYSIS.root/*.root"
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.364704.e7142_s3681_r13144_p6453.FTAG1_TV3_ANALYSIS.root/*.root"
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.364705.e7142_s3681_r13144_p6453.FTAG1_TV3_ANALYSIS.root/*.root"
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.364706.e7142_s3681_r13144_p6453.FTAG1_TV3_ANALYSIS.root/*.root"
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/FTAG1_2025-06-03/user.jecifuen.mc20_13TeV.364707.e7142_s3681_r13144_p6453.FTAG1_TV3_ANALYSIS.root/*.root"
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/JETM2_mc20_alternative_MC_Rafael/Herwing_dipole/*.root"
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/JETM2_mc20_alternative_MC_Rafael/Sherpa_Cluster/*.root"
    "/share/lustre/tmlinare/Lund_tagging/jetmdatamc_output/JETM2_mc20_alternative_MC_Rafael/Sherpa_Lund/*.root"
    "$JETM2_W_flatmass"
)

outfiles=(
    "histos/mass_40-3500_pt_350-3100/topBSMP8_426345.root"
    "histos/mass_40-3500_pt_3500-3100/qcdP8_364703.root"
    "histos/mass_40-3500_pt_3500-3100/qcdP8_364704.root"
    "histos/mass_40-3500_pt_3500-3100/qcdP8_364705.root"
    "histos/mass_40-3500_pt_3500-3100/qcdP8_364706.root"
    "histos/mass_40-3500_pt_3500-3100/qcdP8_364707.root"
    "histos/mass_40-3500_pt_350-3100/qcdHD.root"
    "histos/mass_40-3500_pt_350-3100/qcdSC.root"
    "histos/mass_40-3500_pt_350-3100/qcdSL.root"
    "histos/mass_40-300_pt_200-3100/WBSMP8.root"
    "histos/mass_40-300_pt_200-3100/qcdP8_364703.root"
    "histos/mass_40-300_pt_200-3100/qcdP8_364704.root"
    "histos/mass_40-300_pt_200-3100/qcdP8_364705.root"
    "histos/mass_40-300_pt_200-3100/qcdP8_364706.root"
    "histos/mass_40-300_pt_200-3100/qcdP8_364707.root"
    "histos/mass_40-300_pt_200-3100/qcdHD.root"
    "histos/mass_40-300_pt_200-3100/qcdSC.root"
    "histos/mass_40-300_pt_200-3100/qcdSL.root"
    "histos/mass_40-300_pt_200-3100/WBSMP8_flat.root"
)

signals=(
    top top top top top top top top top
    W W W W W W W W W W
)
# -------------------------------------------------------------

echo "===== [3/8] Array bounds check ====="
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set"; exit 1
fi
idx=$SLURM_ARRAY_TASK_ID
if [ "$idx" -ge "${#infile_lists[@]}" ] || [ "$idx" -ge "${#outfiles[@]}" ] || [ "$idx" -ge "${#signals[@]}" ]; then
  echo "ERROR: Array index $idx out of bounds"; exit 1
fi
echo "Index OK: $idx / ${#infile_lists[@]}"
echo "===================================="

infile_list="${infile_lists[$idx]}"
outfile="${outfiles[$idx]}"
signal="${signals[$idx]}"

# Split the selected infile_list string into patterns
IFS=' ' read -r -a infile_patterns <<< "$infile_list"

echo "===== [4/8] Input summary ====="
echo "signal:  $signal"
echo "outfile: $outfile"
echo "patterns (${#infile_patterns[@]}):"
printf '  - %s\n' "${infile_patterns[@]}"
echo "================================"

# Ensure output directory exists
outdir=$(dirname "$outfile")
mkdir -p "$outdir"

# Expand patterns and check we have files
echo "===== [5/8] Expanding input files ====="
shopt -s nullglob
files=()
for pat in "${infile_patterns[@]}"; do
  for f in $pat; do files+=("$f"); done
done
shopt -u nullglob

echo "Found ${#files[@]} files"
if (( ${#files[@]} == 0 )); then
  echo "ERROR: No files matched the provided patterns."; exit 2
fi
printf '  first few:\n'; printf '    %s\n' "${files[@]:0:5}"
echo "======================================="

echo "===== [6/8] Running make_histos.py ====="
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS  MKL_NUM_THREADS=$MKL_NUM_THREADS"
start_time=$(date +%s)

python make_histos.py \
  --signal "$signal" \
  --infile "${infile_patterns[@]}" \
  --outfile "$outfile" \
  --nbins 100

status=$?
end_time=$(date +%s)
echo "Exit code: $status  Duration: $(( (end_time - start_time)/60 )) min"
echo "========================================"

echo "===== [7/8] Output check ====="
if [ $status -eq 0 ] && [ -s "$outfile" ]; then
  echo "Output exists: $outfile"
  ls -lh "$outfile"
else
  echo "WARNING: Job finished with status $status or output missing/empty."
fi
echo "================================"

echo "===== [8/8] Done ====="
date
