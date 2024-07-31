import torch.nn as nn
import torch
from typing import Tuple
from functools import partial

from cmrxrecon.dl.unet import Unet
from cmrxrecon.dl.sensetivitymodel import SensetivityModel
from cmrxrecon.utils import complex_to_real, root_sum_of_squares, ifft_2d_img, fft_2d_img
from cmrxrecon.metrics import metrics
from pytorch_lightning import LightningModule
from torchvision.utils import make_grid
import einops
import wandb



class VarNetLightning(LightningModule):
    def __init__(self, input_channels: int, cascades:int=4, unet_chans:int=18, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = VarNet(input_channels, cascades = cascades, unet_chans=unet_chans)
        self.loss_fn = lambda x, y: torch.nn.functional.mse_loss(torch.view_as_real(x), torch.view_as_real(y))
        self.automatic_optimization = False
        self.lr = lr

    def training_step(self, batch, batch_index: int): 
        # datashape [b, t, h, w]
        undersampled, fully_sampled, sense = batch
        opt = self.optimizers()
        opt.zero_grad()
         
        split_undersampled = torch.split(undersampled, 3, dim=1)
        fully_sampled_split = torch.split(fully_sampled, 3, dim=1)

        loss_arr = []
        fs_estimates = []
        for under, fs in zip(split_undersampled, fully_sampled_split):
            b, t, c, h, w = under.shape

            under = under.reshape(b*t, 1, c, h, w)
            fs_estimate = self.model(under, under != 0, sense)
            fs_estimate = fs_estimate.reshape(b, t, c, h, w)

            loss = self.loss_fn(fs, fs_estimate)
            self.manual_backward(loss)

            fs_estimates.append(fs_estimate)
            loss_arr.append(loss)

        opt.step() 
        ave_loss = sum(loss_arr)/len(loss_arr)
        fs_estimates = torch.cat(fs_estimates, dim=1)
        self.log('train/loss', ave_loss, prog_bar=True, on_step=True, on_epoch=True)
        if batch_index == 0:
            gt_imgs = root_sum_of_squares(ifft_2d_img(fully_sampled), coil_dim=2)
            gt_imgs = gt_imgs[0, :, :, :].unsqueeze(1).abs()
            grid = make_grid(gt_imgs, normalize=True, value_range=(0, gt_imgs.max()/4))
            self.logger.log_image("train/gt_images", [wandb.Image(grid, caption="Validation Ground Truth Images")])
            # imgs [b, t, h, w]
            es_imgs = root_sum_of_squares(ifft_2d_img(fs_estimate), coil_dim=2)
            es_imgs = es_imgs[0, :, :, :].unsqueeze(1).abs()
            grid = make_grid(es_imgs, normalize=True, value_range=(0, gt_imgs.max()/4))
            self.logger.log_image("train/estimate_images", [wandb.Image(grid, caption="Validation Images")])
        return ave_loss 


    def validation_step(self, batch, batch_index): 
        # datashape [b, t, h, w]
        undersampled, fully_sampled, sense = batch

        b, t, c, h, w = undersampled.shape

        undersampled = undersampled.reshape(b*t, 1, c, h, w)
        fs_estimate = self.model(undersampled, undersampled != 0, sense)
        fs_estimate = fs_estimate.reshape(b, t, c, h, w)

        loss = self.loss_fn(fully_sampled, fs_estimate)

        fs_estimate = root_sum_of_squares(ifft_2d_img(fs_estimate), coil_dim=2)
        fully_sampled = root_sum_of_squares(ifft_2d_img(fully_sampled), coil_dim=2)

        self.log('val/loss', loss, prog_bar=True, on_epoch=True)

        ssim = metrics.calculate_ssim(fully_sampled, fs_estimate, self.device)
        nmse = metrics.calculate_nmse(fully_sampled, fs_estimate)
        psnr = metrics.calculate_psnr(fully_sampled, fs_estimate, self.device)

        self.log_dict(
                {'val/loss': loss, 'val/ssim': ssim, 'val/psnr': psnr, 'val/nmse': nmse},
                on_epoch=True, prog_bar=True, logger=True
                )
        if batch_index == 0:
            gt_imgs = fully_sampled[0, :, :, :].unsqueeze(1).abs()
            grid = make_grid(gt_imgs, normalize=True, value_range=(0, gt_imgs.max()/4))
            self.logger.log_image("train/gt_images", [wandb.Image(grid, caption="Validation Ground Truth Images")])
            # imgs [b, t, h, w]
            es_imgs = fs_estimate[0, :, :, :].unsqueeze(1).abs()
            grid = make_grid(es_imgs, normalize=True, value_range=(0, gt_imgs.max()/4))
            self.logger.log_image("train/estimate_images", [wandb.Image(grid, caption="Validation Images")])

        return {
                'loss': loss, 
                'ssim': ssim, 
                'psnr': psnr, 
                'nmse': nmse
                }


    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_index: int): 
        metrics_dict = self.validation_step(batch, batch_index)

        self.log_dict(
                {'test/loss': metrics_dict['loss'], 'val/ssim': metrics_dict['ssim'], 'val/psnr': metrics_dict['psnr'], 'val/nmse': metrics_dict['nmse']},
                on_epoch=True, prog_bar=True, logger=True
                )
        return metrics_dict


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class VarNet(nn.Module):
    def __init__(self, 
                 input_channels: int,
                 cascades:int = 6,
                 unet_chans: int = 18, 
                 sens_chans:int = 8,
                 ):
        super().__init__()

        # module for cascades
        self.cascade = nn.ModuleList()
        
        # populate cascade with model backbone
        model_backbone = partial(Unet, input_channels, input_channels, chans=unet_chans)
        self.cascades = nn.ModuleList(
            [VarnetBlock(model_backbone()) for _ in range(cascades)]
        )


        # regularizer weight
        self.lambda_reg = nn.Parameter(torch.ones((cascades)))

    # k-space sent in [B, C, H, W]
    def forward(self, reference_k, mask, sense_maps):
        # get sensetivity maps
        assert not torch.isnan(reference_k).any()
        assert not torch.isnan(mask).any()

        assert not torch.isnan(sense_maps).any()

        # current k_space 
        current_k = reference_k.clone()
        for i, cascade in enumerate(self.cascades):
            # go through ith model cascade
            refined_k = cascade(current_k, sense_maps)
            assert not torch.isnan(reference_k).any()
            assert not torch.isnan(refined_k).any()

            data_consistency = mask * (current_k - reference_k)
            # gradient descent step
            current_k = current_k - (self.lambda_reg[i] * data_consistency) - refined_k
        return current_k


class VarnetBlock(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    # sensetivities data [B, contrast, C, H, W]
    def forward(self, images, sensetivities):
        # Reduce
        images = ifft_2d_img(images, axes=[-1, -2])

        # Images now [B, contrast, h, w] (complex)
        images = torch.sum(images * sensetivities.conj(), dim=2)

        # Images now [B, contrast * 2, h, w] (real)
        images = complex_to_real(images)
        images, mean, std = self.norm(images)
        images = self.model(images)
        images = self.unnorm(images, mean, std)
        images = real_to_complex(images)

        # Expand
        images = sensetivities * images[:, :, None, :, :]
        images = fft_2d_img(images, axes=[-1, -2])

        return images
    
    # is this not just instance norm?
    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # instance norm
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-9

        x = (x - mean) / std
        return x, mean, std


    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        x = x * std + mean
        return x

###############################################################################
############# HELPER FUNCTIONS ################################################
###############################################################################

def complex_to_real(images: torch.Tensor):
    assert images.is_complex(), 'Channel dimension should be at least 2'
    # images dims [B, C, H, W, complex]
    images = torch.view_as_real(images)
    images = einops.rearrange(images, 'b c h w cm -> b (cm c) h w')
    return images

def real_to_complex(images: torch.Tensor):
    assert images.shape[1] >= 2, 'Channel dimension should be at least 2'
    images = einops.rearrange(images, 'b (cm c) h w -> b c h w cm', cm=2)
    images = images.contiguous()
    images = torch.view_as_complex(images)
    return images

