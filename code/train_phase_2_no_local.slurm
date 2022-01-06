#!/bin/bash
#SBATCH --job-name=s2_no_local
#SBATCH --ntasks=1
#SBATCH --account=rnt@gpu
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=out_s2_no_local.out
#SBATCH --error=out_s2_no_local.out

echo "Running job on $hostname"
cd /gpfswork/rech/rnt/uuj49ar/Inpainting_project_mva/code

# Load conda env
module load python/3.7.6
conda activate GLCIC

set -x
python train_phase_2_no_local.py /gpfsscratch/rech/rnt/uuj49ar/img_align_celeba  /gpfsscratch/rech/rnt/uuj49ar/inpainting/results/no_local --init_model_cn /gpfsscratch/rech/rnt/uuj49ar/inpainting/results/no_local/phase_1/model_cn_step20000 --data_parallel

