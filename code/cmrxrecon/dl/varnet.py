import torch.nn as nn
import torch
from typing import Tuple
from functools import partial

from cmrxrecon.dl.unet import Unet
from cmrxrecon.dl.sensetivitymodel import SensetivityModel
from cmrxrecon.utils import complex_to_real
from torch.fft import ifftshift, fftshift, fft2, ifft2
from pytorch_lightning import LightningModule
import einops



class VarNetLightning(LightningModule):
    def __init__(self, input_channels: int, cascades:int=4, unet_chans:int=18):
        super().__init__()

        self.model = VarNet(input_channels, cascades = cascades, unet_chans=unet_chans)
        self.loss_fn = lambda x, y: torch.nn.functional.mse_loss(torch.view_as_real(x), torch.view_as_real(y))
        self.automatic_optimization = False

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_index: int): 
        # datashape [b, t, h, w]
        undersampled, fully_sampled = batch
        opt = self.optimizers()
        opt.zero_grad()
         
        split_undersampled = torch.split(undersampled, 3, dim=1)
        fully_sampled_split = torch.split(fully_sampled, 3, dim=1)

        loss_arr = []
        fs_estimates = []
        for under, fs in zip(split_undersampled, fully_sampled_split):
            b, t, c, h, w = under.shape

            under = under.reshape(b*t, 1, c, h, w)
            fs_estimate = self.model(under, under != 0)
            fs_estimate = fs_estimate.reshape(b, t, c, h, w)

            loss = self.loss_fn(fs, fs_estimate)
            self.manual_backward(loss)

            fs_estimates.append(fs_estimate)
            loss_arr.append(loss)

        opt.step() 
        ave_loss = sum(loss_arr)/len(loss_arr)
        self.log('train/loss', ave_loss, prog_bar=True, on_step=True)
        return ave_loss 


    def validation_step(self, batch, batch_index): 
        # datashape [b, t, h, w]
        undersampled, fully_sampled = batch

        b, t, c, h, w = undersampled.shape

        undersampled = undersampled.reshape(b*t, 1, c, h, w)
        fs_estimate = self.model(undersampled, undersampled != 0)
        fs_estimate = fs_estimate.reshape(b, t, c, h, w)

        loss = self.loss_fn(fully_sampled, fs_estimate)
        self.log('val/loss', loss, prog_bar=True, on_epoch=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
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

        # model to estimate sensetivities
        self.sens_model = SensetivityModel(2, 2, chans=sens_chans, mask_center=True)

        # regularizer weight
        self.lambda_reg = nn.Parameter(torch.ones((cascades)))

    # k-space sent in [B, C, H, W]
    def forward(self, reference_k, mask):
        # get sensetivity maps
        assert not torch.isnan(reference_k).any()
        assert not torch.isnan(mask).any()
        sense_maps = self.sens_model(reference_k, mask)

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

fft_2d_img = lambda x, axes: fftshift(ifft2(ifftshift(x, dim=axes), dim=axes), dim=axes)
ifft_2d_img = lambda x, axes: ifftshift(fft2(fftshift(x, dim=axes), dim=axes), dim=axes)
