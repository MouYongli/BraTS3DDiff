#!/bin/bash
nvidia-smi
/work/scratch/sanyal/miniconda3/envs/bratseg/bin/python /home/students/sanyal/Master-Thesis/Projects/BraTS3DDiff/src/train.py experiment=end2end-single_patch_16-random_subset_patches-fixed_eps
