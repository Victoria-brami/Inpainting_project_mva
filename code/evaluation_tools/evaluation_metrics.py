from skimage.metrics import structural_similarity, mean_squared_error, peak_signal_noise_ratio
import numpy as np


def get_mse(gt_loader, gen_loader):
    mse_list = []
    for idx, (gt_batch, gen_batch) in enumerate(zip(gt_loader, gen_loader)):
        mse_list.append(mean_squared_error(gt_batch, gen_batch))
    return np.mean(mse_list)

def get_ssim(gt_loader, gen_loader):
    ssim_list = []
    for idx, (gt_batch, gen_batch) in enumerate(zip(gt_loader, gen_loader)):
        ssim_list.append(structural_similarity(gt_batch, gen_batch))
    return np.mean(ssim_list)

def get_pnsr(gt_loader, gen_loader):
    pnsr_list = []
    for idx, (gt_batch, gen_batch) in enumerate(zip(gt_loader, gen_loader)):
        pnsr_list.append(peak_signal_noise_ratio(gt_batch, gen_batch))
    return np.mean(pnsr_list)
