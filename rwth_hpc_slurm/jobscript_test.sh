#!/usr/bin/zsh

#SBATCH --job-name=patch_sampling-16          # Job name
#SBATCH --output=/hpcwork/ca550013/slurm_logs/tmp/output_%j.txt       # Standard output and error log (%j is job ID)
#SBATCH --error=/hpcwork/ca550013/slurm_logs/tmp/error_%j.txt
#SBATCH --partition=c18m         # Request the GPU partition (adjust as necessary)
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4                 # CPU cores per task (adjust as needed)
#SBATCH --time=04:00:00              # Time limit hrs:min:sec


# Load any necessary modules (e.g., CUDA, Python)

# Run your command (replace with your actual code or script)
echo "Running on $SLURM_JOB_NODELIST with $SLURM_GPUS_PER_NODE GPUs"
/home/ca550013/Miniconda3-New/envs/bratseg/bin/python /home/ca550013/Master-Thesis/Projects/BraTS3DDiff/scripts/patch_selection.py
