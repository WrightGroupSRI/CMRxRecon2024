import einops
import torch
import math

import torch.nn as nn
from cmrxrecon.dl.unet import Unet
from cmrxrecon.utils import ifft_2d_img, complex_to_real, real_to_complex, root_sum_of_squares

from typing import Tuple

class SensetivityModel(nn.Module):
    def __init__(
            self, 
            in_chans: int, 
            out_chans: int, 
            chans: int, 
            mask_center: bool = True, 
            ):
        
        super().__init__()
        self.model = Unet(in_chans, out_chans, chans=chans)
        self.mask_center = mask_center

    # recieve coil maps as [B, contrast, channels, H, W]
    def forward(self, images, mask):
        assert not torch.isnan(images).any()
        images = images[:, [0], :, :, :]
        images = self.mask(images, images != 0) 
        assert not torch.isnan(images).any()
        # get the first image for estimating coil sensetivites

        images = ifft_2d_img(images, axes=[-1, -2])

        number_of_coils = images.shape[2]
        n_time = images.shape[1]

        images = einops.rearrange(images, 'b t c h w -> (b t c) 1 h w')
        assert isinstance(images, torch.Tensor)

        # convert to real numbers [b * contrast * coils, cmplx, h, w]
        images = complex_to_real(images)
        # norm 
        images, mean, std = self.norm(images)
        assert not torch.isnan(images).any()
        # pass through model
        images = self.model(images)
        assert not torch.isnan(images).any()
        # unnorm
        images = self.unnorm(images, mean, std)
        # convert back to complex
        images = real_to_complex(images)
        # rearange back to original format
        images = einops.rearrange(images, '(b t c) 1 h w -> b t c h w', c=number_of_coils, t=n_time)
        # rss to normalize sense maps
        rss_norm = root_sum_of_squares(images, coil_dim=2).unsqueeze(2) + 1e-9
        #assert not (rss_norm == 0).any()
        images = images / rss_norm
        return images

    def mask(self, coil_k_spaces, center_mask):
        # coil_k: [b cont chan height width]
        center_x = center_mask.shape[-1] // 2
        center_y = center_mask.shape[-2] // 2

        mask = torch.zeros_like(coil_k_spaces, dtype=torch.bool)
        mask[:, :, :, center_y-8:center_y+8, center_x-8:center_x+8] = 1
        
        return coil_k_spaces * mask 


    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1) + 1e-6

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean
