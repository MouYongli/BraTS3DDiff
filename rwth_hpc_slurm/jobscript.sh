#!/usr/bin/zsh

#SBATCH --job-name=bratseg_sampling_inter_res_mergejob           # Job name
#SBATCH --output=/home/ca550013/Master-Thesis/Projects/BraTS3DDiff/slurm/output_%j.txt       # Standard output and error log (%j is job ID)
#SBATCH --error=/home/ca550013/Master-Thesis/Projects/BraTS3DDiff/slurm/error_%j.txt
#SBATCH --nodes=1                    # Run on a single node
#SBATCH --ntasks=2                   # Run a single task
#SBATCH --gpus=2                     # Request 2 GPUs
#SBATCH --time=02:00:00              # Time limit hrs:min:sec
#SBATCH --partition=c18g              # Request the GPU partition (adjust as necessary)
#SBATCH --cpus-per-task=8                 # CPU cores per task (adjust as needed)

# Load any necessary modules (e.g., CUDA, Python)
module load cuda/12.4                # Example module for CUDA (modify as necessary)
module load miniconda3               # Example module for Python

eval "$(conda shell.bash hook)"
# Activate your virtual environment if needed
conda activate bratseg

# Run your command (replace with your actual code or script)
echo "Running on $SLURM_JOB_NODELIST with $SLURM_GPUS_PER_NODE GPUs"
/home/ca550013/Miniconda3-New/envs/bratseg/bin/python /home/ca550013/Master-Thesis/Projects/BraTS3DDiff/src/eval.py
conda deactivate
