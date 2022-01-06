#!/bin/bash
#SBATCH --job-name=no_local
#SBATCH --partition=gpu_p2
#SBATCH --ntasks=1
#SBATCH --account=rnt@gpu
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=out_no_local.out
#SBATCH --error=out_no_local.out

echo "Running job on $hostname"
cd /gpfswork/rech/rnt/uuj49ar/Inpainting_project_mva/code

# Load conda env
module load python/3.7.6
conda activate GLCIC

set -x
python train_no_local.py /gpfsscratch/rech/rnt/uuj49ar/img_align_celeba  /gpfsscratch/rech/rnt/uuj49ar/inpainting/results/no_local --data_parallel 
