#!/bin/bash
nvidia-smi
/work/scratch/sanyal/miniconda3/envs/bratseg/bin/python /home/students/sanyal/Master-Thesis/Projects/BraTS3DDiff/src/train.py experiment=patch_diffusion_end2end_one_patch_random_subset_patches_16_unet_quarter
