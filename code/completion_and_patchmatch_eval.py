import torch
from code.evaluation_tools.evaluate import ModelEvaluation
from torch.utils.data import DataLoader
from code.tools.fixseed import fixseed
from code.tools.metric_tools import format_metrics, save_metrics
from tqdm import tqdm
from code.utils import gen_input_mask, gen_hole_area
from code.models import CompletionNetwork, CompletionNetworkZero
import numpy as np
from PIL import Image
from code.comparison_datasets import ComparisonImageDataset
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset


class ComparisonEvalDataset(Dataset):

    def __init__(self, mode, data_iterator, params):
        self.batches = []
        self.transforms = transforms.Compose(
            [transforms.Resize((299, 299)), transforms.ToPILImage(), transforms.ToTensor()])
        self.transforms = None
        self.data_iterator = data_iterator
        self.params = params
        self.mode = mode
        assert self.mode in ["cn", "patch7", "gt"]
        self.build_batches()

    def build_batches(self):
        with torch.no_grad():
            for databatch in tqdm(self.data_iterator, desc=f"Construct EVAL dataset: {self.mode}.."):
                batch = databatch.to(self.params["device"])
                batch_size = batch.shape[0]
                for i in range(batch_size):
                    self.batches.append(batch[i, :, :, :])
        del self.data_iterator

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, item):
        sample = self.batches[item]
        if self.transforms:
            sample = self.transforms(sample)
        # print('\n Considering a sample of size .....', sample.shape)
        return sample


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


def comparison_evaluate(params, device='cpu', fid_only=False):
    print("0_ Start evaluation process")
    model_evaluator = ModelEvaluation(device=device)

    # load dataset
    trnsfm = transforms.Compose([
        #transforms.Resize(params["cn_input_size"]),
        # transforms.RandomCrop((params["cn_input_size"], params["cn_input_size"])),
        transforms.ToTensor(),
    ])
    print('3_ loading dataset... (it may take a few minutes)')
    gt1_dataset = ComparisonImageDataset(os.path.join(params["comparison_data_dir"]),
                               transform=trnsfm,
                               recursive_search=True, image_type='gt')
    gt2_dataset = ComparisonImageDataset(os.path.join(params["comparison_data_dir"]),
                               transform=trnsfm,
                               recursive_search=True, image_type=params["model_to_compare"])
    mask_dataset = ComparisonImageDataset(os.path.join(params["comparison_data_dir"]),
                                         transform=None,
                                         recursive_search=True, image_type='mask')
    all_seeds = list(range(params["niter"]))
    all_metrics = {}

    print(params.keys())
    params["mpv"] = compute_mpv(gt1_dataset, None, device="cpu")

    try:
        for (idx, seed) in enumerate(all_seeds):
            print(f"Evaluation number: {idx + 1}/{params['niter']}")
            fixseed(seed)

            # gt1_dataset.reset_shuffle()
            #gt1_dataset.shuffle()
            # gt2_dataset.reset_shuffle()
            #gt2_dataset.shuffle()

            data_iterator = DataLoader(gt1_dataset, batch_size=params["comparison_batch_size"], shuffle=False, num_workers=2)
            data_iterator2 = DataLoader(gt2_dataset, batch_size=params["comparison_batch_size"], shuffle=False,
                                       num_workers=2)
            gt_dataset = ComparisonEvalDataset("gt", data_iterator, params)
            print(params["model_to_compare"])
            gen_dataset =  ComparisonEvalDataset(params["model_to_compare"], data_iterator2, params)

            gt_loader = DataLoader(gt_dataset, batch_size=params["comparison_batch_size"], shuffle=False, num_workers=2)
            gen_loader = DataLoader(gen_dataset, batch_size=params["comparison_batch_size"], shuffle=False, num_workers=2)

            all_metrics[seed] = model_evaluator.evaluate_model(gt_loader, gen_loader, mask_dataset, fid_only=False)

            del gt_loader
            del gen_loader
            del gen_dataset
            del gt_dataset

    except KeyboardInterrupt:
        string = "Saving the evaluation before exiting.."
        print(string)

    epoch = params["checkpointpath"].split("_")[-1][4:]
    output_folder = '../evaluation/'

    metricname = "evaluation_metrics_{}_{}_comparison.yaml".format(epoch, params["model_to_compare"])

    metrics = {"feats": {key: [format_metrics(all_metrics[seed])[key] for seed in all_metrics.keys()] for key in
                         all_metrics[all_seeds[0]]}}

    evalpath = os.path.join(output_folder, metricname)
    print(f"Saving evaluation in: {evalpath}")
    save_metrics(evalpath, metrics)
