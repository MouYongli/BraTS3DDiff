#!/bin/bash
source /work/scratch/sanyal/miniconda3/etc/profile.d/conda.sh
nvidia-smi
conda activate bratseg
python /home/students/sanyal/Master-Thesis/Projects/BraTS3DDiff/src/train_classify.py
conda deactivate

