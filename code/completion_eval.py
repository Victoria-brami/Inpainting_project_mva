import torch
from code.evaluation_tools.evaluate import ModelEvaluation
from torch.utils.data import DataLoader
from code.tools.fixseed import fixseed
from code.tools.metric_tools import format_metrics, save_metrics
from tqdm import tqdm
from code.utils import gen_input_mask, gen_hole_area
from code.models import CompletionNetwork
import numpy as np
from PIL import Image
from code.datasets import ImageDataset
import torchvision.transforms as transforms
import os

class EvalDataloader(object):

    def __init__(self, mode, model, data_iterator, params):
        assert mode in ["gen", "rc", "gt"]
        self.batches = []
        self.transforms = transforms.Compose([transforms.Resize(299), transforms.ToPILImage(), transforms.ToTensor()])
        with torch.no_grad():
            for databatch in tqdm(data_iterator, desc=f"Construct dataloader: {mode}.."):
                if mode == "gt":
                    batch = databatch.to(params["device"])
                else:
                    databatch = databatch.to(params["device"])
                    mask = gen_input_mask(
                        shape=(databatch.shape[0], 1, databatch.shape[2], databatch.shape[3]),
                        hole_size=(
                            (params["hole_min_w"], params["hole_max_w"]),
                            (params["hole_min_h"], params["hole_max_h"])),
                        hole_area=gen_hole_area(
                            (params["ld_input_size"], params["ld_input_size"]),
                            (databatch.shape[3], databatch.shape[2])),
                        max_holes=params["max_holes"],
                    ).to(params["device"])
                    x_mask = databatch - databatch * mask + params["mpv"] * mask
                    input = torch.cat((x_mask, mask), dim=1)
                    batch = model(input)
                self.batches.append(self.transforms(batch))

    def __iter__(self):
        return iter(self.batches)



def compute_mpv(train_dataset, mpv=None, device="cpu"):
    # compute mpv (mean pixel value) of training dataset
    if mpv is None:
        mpv = np.zeros(shape=(3,))
        pbar = tqdm(total=len(train_dataset.imgpaths), desc='computing mean pixel value of training dataset...')
        for imgpath in train_dataset.imgpaths:
            img = Image.open(imgpath)
            x = np.array(img) / 255.
            mpv += x.mean(axis=(0, 1))
            pbar.update()
        mpv /= len(train_dataset.imgpaths)
        pbar.close()
    else:
        mpv = np.array(mpv)
    mpv = torch.tensor(
        mpv.reshape(1, 3, 1, 1),
        dtype=torch.float32).to(device)
    return mpv


def evaluate(params, device='cpu'):
    print("0_ Start evaluation process")
    model = CompletionNetwork()
    state_dict = torch.load(params["checkpointpath"], map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model_evaluator = ModelEvaluation(device=device)

    # load dataset
    trnsfm = transforms.Compose([
        transforms.Resize(params["cn_input_size"]),
        #transforms.RandomCrop((params["cn_input_size"], params["cn_input_size"])),
        transforms.ToTensor(),
    ])
    print('3_ loading dataset... (it may take a few minutes)')
    gt1_dataset = ImageDataset(os.path.join(params["data_dir"], 'train'),
                              trnsfm,
                              recursive_search=params["recursive_search"])
    gt2_dataset = ImageDataset(os.path.join(params["data_dir"], 'train'),
                              trnsfm,
                              recursive_search=params["recursive_search"])
    all_seeds = list(range(params["niter"]))
    all_metrics = {}

    print(params.keys())
    params["mpv"] = compute_mpv(gt1_dataset, None, device="cpu")

    try:
        for (idx, seed) in enumerate(all_seeds):
            print(f"Evaluation number: {idx + 1}/{params['niter']}")
            fixseed(seed)

            # gt1_dataset.reset_shuffle()
            gt1_dataset.shuffle()
            # gt2_dataset.reset_shuffle()
            gt2_dataset.shuffle()

            data_iterator = DataLoader(gt1_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=2)
            gt_loader = EvalDataloader("gt", model, data_iterator, params)
            gen_loader = EvalDataloader("gen", model, data_iterator, params)

            all_metrics[seed] = model_evaluator.evaluate_model(gt_loader, gen_loader)

            print("Iteration {}: FID score: {}".format(idx, all_metrics[seed]))

    except KeyboardInterrupt:
        string = "Saving the evaluation before exiting.."
        print(string)

    epoch = params["checkpointpath"].split("_")[-1][4:]
    output_folder = os.path.join(params["checkpointpath"].split("/")[:-1])

    metricname = "evaluation_metrics_{}_all.yaml".format(epoch)

    metrics = {"feats": {key: [format_metrics(all_metrics[seed])[key] for seed in all_metrics.keys()] for key in all_metrics[all_seeds[0]]}}
    
    evalpath = os.path.join(output_folder, metricname)
    print(f"Saving evaluation: {evalpath}")
    save_metrics(evalpath, metrics)
