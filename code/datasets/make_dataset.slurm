#!/bin/bash
#SBATCH --job-name=make_dataset
#SBATCH --partition=gpu_p2
#SBATCH --ntasks=1
#SBATCH --account=rnt@gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=1:00:00
#SBATCH --output=out_make.out
#SBATCH --error=out_make.out

echo "Running job on $hostname"
cd /gpfswork/rech/rnt/uuj49ar/Inpainting_project_mva/author_code/datasets

# Load conda env
module load python/3.7.6
conda activate GLCIC

set -x

python make_dataset.py /gpfsscratch/rech/rnt/uuj49ar/img_align_celeba

