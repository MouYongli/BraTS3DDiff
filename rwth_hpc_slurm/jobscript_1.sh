#!/usr/bin/zsh


#SBATCH --job-name=bratseg_sampling_inter_res_mergejob-1           # Job name
#SBATCH --output=/hpcwork/ca550013/slurm_logs/tmp/output_%j.txt       # Standard output and error log (%j is job ID)
#SBATCH --error=/hpcwork/ca550013/slurm_logs/tmp/error_%j.txt
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4                 # CPU cores per task (adjust as needed)
#SBATCH --time=04:00:00              # Time limit hrs:min:sec


task_name="predict"
task_tag="BraTS23-Patch-Diffusion-512-model-inter-res-merge-1"
now=$(date +"%y-%m-%d-%H-%M-%S")
logdir="/hpcwork/ca550013/slurm_logs/log/${task_tag}/${task_name}/${now}"
mkdir -p $logdir

# Load any necessary modules (e.g., CUDA, Python)
#module load CUDA/12.4.0                # Example module for CUDA (modify as necessary)

# Activate your virtual environment if needed
source /home/ca550013/Miniconda3-New/bin/activate bratseg

# Run your command (replace with your actual code or script)
echo "Running job $SLURM_JOB_ID on $SLURM_JOB_NODELIST with $SLURM_GPUS_PER_NODE GPUs"
srun python /home/ca550013/Master-Thesis/Projects/BraTS3DDiff/src/eval.py

source /home/ca550013/Miniconda3-New/bin/deactivate

#move tmp output files to logdir
mv /hpcwork/ca550013/slurm_logs/tmp/output_$SLURM_JOB_ID.txt $logdir
mv /hpcwork/ca550013/slurm_logs/tmp/error_$SLURM_JOB_ID.txt $logdir

echo "Finished Job $SLURM_JOB_ID"



