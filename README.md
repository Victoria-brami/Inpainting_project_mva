# Inpainting_project_mva

## 0. Overview
This project studies the inpainting method described in the article [Globally and Locally Consistent Image Completion](http://iizuka.cs.tsukuba.ac.jp/projects/completion/en/).
The repo is largely inspired from [this project](https://github.com/otenim/GLCIC-PyTorch.git).

## 1. Requirements
Run this line to install all the needed requirements.
```
pip install -r requirements.txt
```
You must also retrieve

## 2. Run the code
CelebA download link is provided in the [project repo](https://github.com/otenim/GLCIC-PyTorch.git).
### Training Phases

Train without global discriminator:
```
cd code
python train_phase_2_no_global.py  [PATH TO DATASET FOLDER]  [OUTPUT-DIR To store checkpoint] --init_model_cn [CHECKPOINT PHASE 1] --data_parallel
python train_phase_3_no_global.py  [PATH TO DATASET FOLDER]  [OUTPUT-DIR To store checkpoint] --init_model_cn [CHECKPOINT PHASE 2] --data_parallel
```

Train without local discriminator:
```
cd code
python train_phase_2_no_local.py  [PATH TO DATASET FOLDER]  [OUTPUT-DIR To store checkpoint] --init_model_cn [CHECKPOINT PHASE 1] --data_parallel
python train_phase_3_no_local.py  [PATH TO DATASET FOLDER]  [OUTPUT-DIR To store checkpoint] --init_model_cn [CHECKPOINT PHASE 1] --data_parallel
```

### Evaluation Phases
All the evaluations files (metric computation) are stored in `evaluation_tools` folder and files including `èval` in their name.

##### Evaluate on the whole CelebA test set
You can run for discriminants ablation study case:
```
cd code
python main_eval.py  --data_dir [PATH TO DATA FOLDER] --checkpointpath [PATH TO CHECKPOINT DEPENDING ON THE TRAINING (with/without discriminator)
```
When studying inner parameters ablation:
```
cd code
python main_eval.py  --data_dir [PATH TO DATA FOLDER] --checkpointpath [PATH TO CHECKPOINT DEPENDING ON THE TRAINING] --layer_idx [LAYER NUMBER] --seed [RANDOM INT]
```
We use different seeds to always erase different channels in a same layer.

##### Evaluate on the batch of 280 images of CelebA for comparison
You may run :
```
cd code
python main_comparison_eval.py --comparison_data_dir [PATH TO DATA FOLDER] --model_to_compare [MODEL USED] --output_folder [WHERE THE SCORES ARE SAVED]
```
With MODEL USED being either 'cn', 'channel44_cn', 'no_local_cn', 'patch7', 'no_global_cn', 'channel0_cn'.

### Prediction
```
cd code
python predict.py [PATH TO MODEL CHECKPOINT] config.json  [IMAGE] [NAME OF THE OUTPUT IMAGE]
```