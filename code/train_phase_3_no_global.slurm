#!/bin/bash
#SBATCH --job-name=s3_no_global
#SBATCH --ntasks=1
#SBATCH --account=rnt@gpu
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=out_s3_no_global.out
#SBATCH --error=out_s3_no_global.out

echo "Running job on $hostname"
cd /gpfswork/rech/rnt/uuj49ar/Inpainting_project_mva/code

# Load conda env
module load python/3.7.6
conda activate GLCIC

set -x
python train_phase_3_no_global.py /gpfsscratch/rech/rnt/uuj49ar/img_align_celeba  /gpfsscratch/rech/rnt/uuj49ar/inpainting/results/no_global --init_model_cn /gpfsscratch/rech/rnt/uuj49ar/inpainting/results/no_global/phase_1/model_cn_step90000  --init_model_cd /gpfsscratch/rech/rnt/uuj49ar/inpainting/results/no_global/phase_2_only/model_cd_step10000 --data_parallel
