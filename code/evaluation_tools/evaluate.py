import torch
import torch.nn as nn
import numpy as np
from code.evaluation_tools.fid import get_fid
from code.evaluation_tools.evaluation_metrics import get_mse, get_ssim, get_pnsr
import torchvision

class FIDInceptionV3(nn.Module):

    def __init__(self):
        super(FIDInceptionV3, self).__init__()
        resnet_model = torchvision.models.resnet50(pretrained=False)
        #checkpoint = torch.load('/gpfswork/rech/rnt/uuj49ar/resnet50-0676ba61.pth')
        checkpoint = torch.load('../../resnet50-0676ba61.pth', map_location='cpu')
        resnet_model.load_state_dict(checkpoint)

        self.model = torch.nn.Sequential(*list(resnet_model.children())[:-1])

    def forward(self, x):
        # reshape x
        #print('Input shape is :', x.shape)
        x = self.model(x)
        return x


class ModelEvaluation:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = FIDInceptionV3()
        self.model.eval()
        print("Inception model loaded")

    def compute_embeddings(self, loader):
        activations = []
        with torch.no_grad():
            for (idx, batch) in enumerate(loader):
                #print(" Batch input shape: ", batch.shape)
                activations.append(self.model(batch))
            activations = torch.cat(activations, dim=0)
        return activations

    @staticmethod
    def compute_activations_statistics(activations):
        activations = activations.cpu().numpy()
        #print("\n Shape of the activations", type(activations), activations.shape, activations[:, :, 0, 0].shape)
        activations = activations[:, :, 0, 0]
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def evaluate_model(self, gt_loader, gen_loader, mask_loader=None, fid_only=True):
        metrics = {}
        gt_features = self.compute_embeddings(gt_loader)
        gen_features = self.compute_embeddings(gen_loader)
        gt_stats = self.compute_activations_statistics(gt_features)
        gen_stats = self.compute_activations_statistics(gen_features)
        gt_params = {"mu": gt_stats[0], "sigma": gt_stats[1]}
        gen_params = {"mu": gen_stats[0], "sigma": gen_stats[1]}

        metrics["fid"] = get_fid(gt_params, gen_params)

        if not fid_only:
            metrics["mse"] = get_mse(gt_loader, gen_loader, mask_loader)
            metrics["ssim"] = get_ssim(gt_loader, gen_loader, mask_loader)
            metrics["pnsr"] = get_pnsr(gt_loader, gen_loader, mask_loader)

        return metrics
