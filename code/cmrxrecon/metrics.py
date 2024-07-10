import torch
import pytorch_lightning as pl
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio


"""
Usage

metrics = metrics()

ssim_value = metrics.calculate_ssim(ground_truth, predicted)
psnr_value = metrics.calculate_psnr(ground_truth, predicted)
nmse_value = metrics.calculate_nmse(ground_truth, predicted)


"""

class metrics(pl.LightningModule):

    def __init__(self):
        super().__init__()

    def calculate_ssim(self, ground_truth, predicted):
        """
        Calculate the Structural Similarity Index Measure (SSIM) between ground truth and predicted images.
        
        Args:
        ground_truth (torch.Tensor): The ground truth image tensor.
        predicted (torch.Tensor): The predicted image tensor.

        Returns:
        torch.Tensor: The SSIM value.
        """
        data_range = ground_truth.max() - ground_truth.min()
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range.item())
        return ssim_metric(ground_truth, predicted)

    def calculate_psnr(self, ground_truth, predicted):
        """
        Calculate the Peak Signal-to-Noise Ratio (PSNR) between ground truth and predicted images.
        
        Args:
        ground_truth (torch.Tensor): The ground truth image tensor.
        predicted (torch.Tensor): The predicted image tensor.

        Returns:
        torch.Tensor: The PSNR value.
        """
        data_range = ground_truth.max() - ground_truth.min()
        psnr_metric = PeakSignalNoiseRatio(data_range=data_range.item())
        return psnr_metric(ground_truth, predicted)

    def calculate_nmse(self, ground_truth, predicted):
        """
        Calculate the Normalized Mean Squared Error (NMSE) between ground truth and predicted images.
        
        Args:
        ground_truth (torch.Tensor): The ground truth image tensor.
        predicted (torch.Tensor): The predicted image tensor.

        Returns:
        torch.Tensor: The NMSE value.
        """
        mse = torch.mean((ground_truth - predicted) ** 2)
        norm = torch.mean(ground_truth ** 2)
        return mse / norm
