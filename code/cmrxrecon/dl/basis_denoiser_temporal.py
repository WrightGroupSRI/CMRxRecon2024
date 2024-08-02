import torch.nn as nn
import torch
from typing import Tuple
from functools import partial
import wandb

from cmrxrecon.dl.unet import Unet
from cmrxrecon.dl.resnet import ResNet
from cmrxrecon.metrics import metrics
from cmrxrecon.utils import ifft_2d_img
from cmrxrecon.dl.lowrank_varnet import view_as_complex, view_as_real
from torch.fft import ifftshift, fftshift, fft2, ifft2
import pytorch_lightning as pl 
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import matplotlib



class TemporalDenoiser(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = ResNet(in_chan=6, out_chan=6, chans=18, dimension='1d')
        self.loss_fn = lambda x, y: torch.nn.functional.l1_loss((x), (y))
        matplotlib.use('Agg')

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_index: int): 
        undersampled, fully_sampled, sense = batch
        
        masked_k = self.get_center_masked_k_space(undersampled) 
        masked_k = (ifft_2d_img(masked_k) * sense.conj()).sum(2) / (sense * sense.conj() + 1e-6).sum(2)
        temporal_basis, _ = self.get_singular_vectors(masked_k)

        fully_sampled_images = (ifft_2d_img(fully_sampled) * sense.conj()).sum(2) / (sense * sense.conj()).sum(2)
        gt_temporal_basis, _ = self.get_singular_vectors(fully_sampled_images)
        gt_temporal_basis = view_as_real(gt_temporal_basis.transpose(-1, -2)).transpose(-1, -2)
        

        temporal_basis = temporal_basis.transpose(-1, -2)
        temporal_basis = view_as_real(temporal_basis) 
        denoised_temporal = temporal_basis + self.model((temporal_basis))
        denoised_temporal = denoised_temporal.transpose(-1, -2)
        
        loss = self.loss_fn(denoised_temporal, gt_temporal_basis)

        self.log('train/loss', loss, on_step=True, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        
        if batch_index == 0:  # Log only for the first batch in each epoch
            with torch.no_grad():
                gt_temporal_basis = view_as_complex(gt_temporal_basis.transpose(-1, -2)).transpose(-1, -2).cpu().detach()
                denoised_temporal = view_as_complex(denoised_temporal.transpose(-1, -2)).transpose(-1, -2).cpu().detach()
                plt.clf()
                plt.figure(figsize=(10, 10))
                for i in range(denoised_temporal.shape[2]):
                    plt.plot(denoised_temporal[0, :, i].abs(), label=f'Estimated {i}', color='orange') 
                for i in range(denoised_temporal.shape[2]):
                    plt.plot(gt_temporal_basis[0, :, i].abs(), label=f'Ground Truth {i}', color='blue') 
                plt.legend()

                self.logger.log_image("train/singular", [wandb.Image(plt)])

        return loss


    def validation_step(self, batch, batch_index): 
        undersampled, fully_sampled, sense = batch
        
        masked_k = self.get_center_masked_k_space(undersampled) 
        masked_k = (ifft_2d_img(masked_k) * sense.conj()).sum(2) / (sense * sense.conj() + 1e-6).sum(2)
        temporal_basis, _ = self.get_singular_vectors(masked_k)

        fully_sampled_images = (ifft_2d_img(fully_sampled) * sense.conj()).sum(2) / (sense * sense.conj()).sum(2)
        gt_temporal_basis, _ = self.get_singular_vectors(fully_sampled_images)
        gt_temporal_basis = view_as_real(gt_temporal_basis.transpose(-1, -2)).transpose(-1, -2)

        temporal_basis = temporal_basis.transpose(-1, -2)
        temporal_basis = view_as_real(temporal_basis)
        denoised_temporal = temporal_basis + self.model((temporal_basis))
        denoised_temporal = denoised_temporal.transpose(-1, -2)
        
        loss = self.loss_fn(denoised_temporal, gt_temporal_basis)

        self.log('val/loss', loss, on_step=True, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        
        if batch_index == 0:  # Log only for the first batch in each epoch
            with torch.no_grad():
                gt_temporal_basis = view_as_complex(gt_temporal_basis.transpose(-1, -2)).transpose(-1, -2).cpu().detach()
                denoised_temporal = view_as_complex(denoised_temporal.transpose(-1, -2)).transpose(-1, -2).cpu().detach()
                plt.clf()
                plt.figure(figsize=(10, 10))
                for i in range(denoised_temporal.shape[2]):
                    plt.plot(denoised_temporal[0, :, i].abs(), label=f'Estimated {i}', color='orange') 
                for i in range(denoised_temporal.shape[2]):
                    plt.plot(gt_temporal_basis[0, :, i].abs(), label=f'Ground Truth {i}', color='blue') 

                plt.legend()

                self.logger.log_image("val/singular", [wandb.Image(plt)])

        return loss

    def test_step(self, batch, batch_index): 
        undersampled, fully_sampled, sense = batch
        
        masked_k = self.get_center_masked_k_space(undersampled) 
        masked_k = (ifft_2d_img(masked_k) * sense.conj()).sum(2) / (sense * sense.conj() + 1e-6).sum(2)
        temporal_basis, _ = self.get_singular_vectors(masked_k)

        fully_sampled_images = (ifft_2d_img(fully_sampled) * sense.conj()).sum(2) / (sense * sense.conj() + 1e-6).sum(2)
        gt_temporal_basis, _ = self.get_singular_vectors(fully_sampled_images)
        gt_temporal_basis = view_as_real(gt_temporal_basis.transpose(-1, -2)).transpose(-1, -2)

        temporal_basis = temporal_basis.transpose(-1, -2)
        temporal_basis = view_as_real(temporal_basis)
        denoised_temporal = temporal_basis + self.model((temporal_basis))
        denoised_temporal = denoised_temporal.transpose(-1, -2)
        
        loss = self.loss_fn(denoised_temporal, gt_temporal_basis)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def rss(self, data):
        return ifft_2d_img(data).abs().pow(2).sum(2).sqrt()
    
    def prepare_images(self, imgs, max_val):
        imgs = imgs[0, :, :, :].unsqueeze(1)
        grid = make_grid(imgs, normalize=True, value_range=(0, max_val), nrow=3)
        return grid

    def get_singular_vectors(self, data):
        b, t, h, w = data.shape

        temporal_basis, _, spatial_basis = torch.svd(data.view(b, t, h*w))
        spatial_basis = spatial_basis.conj().transpose(-1, -2)
        components = 3
        temporal_basis = temporal_basis[:, :, :components]
        spatial_basis = spatial_basis[:, :components, :]
        spatial_basis = spatial_basis.view(b, components, h, w)
        
        return temporal_basis, spatial_basis
    
    def get_center_masked_k_space(self, k_space):
        center_mask = torch.zeros_like(k_space, dtype=torch.bool)
        center_y, center_x = center_mask.shape[-2:]
        center_mask[:, :, :, center_y//2-8:center_y//2+8, center_x//2-8:center_x//2+8] = 1
        masked_k = k_space * center_mask
        return masked_k

    def combined_bases(self, spatial_basis, temporal_basis):
        b, components, h, w = spatial_basis.shape
        b, t, components = temporal_basis.shape
        images = (temporal_basis @ spatial_basis.reshape(b, components, h*w)).view(b, t, h, w)
        return images

