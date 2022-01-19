#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=gpu_p2
#SBATCH --ntasks=1
#SBATCH --account=rnt@gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=out/eval.out
#SBATCH --error=out/eval.out

echo "Running job on $hostname"
cd /gpfswork/rech/rnt/uuj49ar/Inpainting_project_mva

# Load conda enc
module purge
module load python/3.7.6
conda init
conda activate GLCIC
export PYTHONPATH="$WORK/Inpainting_project_mva"


python code/main_eval.py --niter 5 --batch_size 64
