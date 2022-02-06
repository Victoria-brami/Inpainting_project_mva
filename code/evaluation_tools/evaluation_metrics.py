from skimage.metrics import structural_similarity, mean_squared_error, peak_signal_noise_ratio
import numpy as np
import torch

def get_mse(gt_loader, gen_loader, mask_list=None):
    mse_list = []
    counter = 0
    with torch.no_grad():
        for idx, (gt_batch, gen_batch) in enumerate(zip(gt_loader, gen_loader)):
            batch_size = gt_batch.shape[0]
            if batch_size != 0:
                if type(gt_batch) == torch.Tensor or type(gen_batch) == torch.Tensor:
                    gt_batch = gt_batch.permute(0, 2, 3, 1).numpy()
                    gen_batch = gen_batch.permute(0, 2, 3, 1).numpy()
                for img_idx in range(batch_size):
                    counter += 1
                    mse_list.append(mean_squared_error(gt_batch[img_idx, :, :, :], gen_batch[img_idx, :, :, :]))
    print("MSE Done")
    return np.mean(mse_list)

def get_ssim(gt_loader, gen_loader, mask_loader):
    ssim_list = []
    counter = 0
    with torch.no_grad():
        for idx, (gt_batch, gen_batch) in enumerate(zip(gt_loader, gen_loader)):
            batch_size = gt_batch.shape[0]
            if batch_size != 0:
                if type(gt_batch) == torch.Tensor or type(gen_batch) == torch.Tensor:
                    gt_batch = gt_batch.permute(0, 2, 3, 1).numpy()
                    gen_batch = gen_batch.permute(0, 2, 3, 1).numpy()
                for img_idx in range(batch_size):
                    im_gt = gt_batch[img_idx,:,:,:]
                    im_gen = gen_batch[img_idx,:,:,:]
                    im_gt = im_gt[np.where(np.array(mask_loader[counter]) != 0)]
                    im_gen = im_gen[np.where(np.array(mask_loader[counter]) != 0)]
                    print(np.array(mask_loader[counter]).shape, im_gt.shape, im_gen.shape)
                    ssim_list.append(structural_similarity(im_gt, im_gen, multichannel=True))
                    #ssim_list.append(structural_similarity(gt_batch[img_idx, :, :, :]*mask_loader[counter], gen_batch[img_idx, :, :, :]*mask_loader[counter, :, :], multichannel=True))
                    counter +=1
    print("SSIM Done")
    return np.mean(ssim_list)

def get_pnsr(gt_loader, gen_loader, mask_list=None):
    pnsr_list = []
    with torch.no_grad():
        for idx, (gt_batch, gen_batch) in enumerate(zip(gt_loader, gen_loader)):
            batch_size = gt_batch.shape[0]
            if batch_size != 0:
                if type(gt_batch) == torch.Tensor or type(gen_batch) == torch.Tensor:
                    gt_batch = gt_batch.permute(0, 2, 3, 1).numpy()
                    gen_batch = gen_batch.permute(0, 2, 3, 1).numpy()
                for img_idx in range(batch_size):
                    pnsr_list.append(peak_signal_noise_ratio(gt_batch[img_idx, :, :, :], gen_batch[img_idx, :, :, :]))
    print("PNSR Done")
    return np.mean(pnsr_list)
