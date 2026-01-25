# setup for lxplus-gpu
if hostnamectl | grep -q "Red Hat Enterprise Linux 9"; then
    # set up the LCG release LCG_104cuda if the highest supported CUDA version is >= 11.8
    cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | cut -d"." -f1,2)
    if (( $(echo "$cuda_version >= 11.8" | bc -l) )); then
        echo "sourcing /cvmfs/sft.cern.ch/lcg/views/LCG_104cuda/x86_64-el9-gcc11-opt/setup.sh"
        source /cvmfs/sft.cern.ch/lcg/views/LCG_104cuda/x86_64-el9-gcc11-opt/setup.sh
    fi

# setup for UCL Hypatia GPU partition
elif [[ $(hostname) == compute-gpu-0-*.local ]]; then
    source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
    conda activate /share/rcifdata/tmlinare/conda/envs/pytorch_py39_cu126

# setup for UCL HEP GPU server
elif [ $(hostname) == "dias.hpc.phys.ucl.ac.uk" ]; then
    # set up conda
    eval "$('/mnt/storage/tmlinare/installs/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    . "/mnt/storage/tmlinare/installs/miniforge3/etc/profile.d/mamba.sh"
    # activate conda environment
    conda activate /mnt/storage/tmlinare/conda/envs/pytorch_py39_cu102

# setup for CentOS 7 machines with CVMFS access, no GPU
elif hostnamectl | grep -q "CentOS Linux 7"; then
    echo "sourcing /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-centos7-gcc12-opt/setup.sh"
    source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-centos7-gcc12-opt/setup.sh

else
    echo "No setup configured for your system."
fi