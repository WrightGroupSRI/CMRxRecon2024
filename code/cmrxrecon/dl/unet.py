import math
from typing import Tuple, List

import torch.nn as nn 
import wandb
import torch
import torch.nn.functional as F
import pytorch_lightning as pl 
from ..utils import root_sum_of_squares, ifft_2d_img
from ..metrics import metrics
from torchvision.utils import make_grid

class UnetLightning(pl.LightningModule):
    def __init__(self, input_channels: int, depth:int=4, chan:int=18, lr:float=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = Unet(input_channels, input_channels, depth, chan)
        self.lr = lr


    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_index: int): 
        undersampled, fully_sampled, _ = batch

        b, t, c, y, x = undersampled.shape
        aliased = root_sum_of_squares(ifft_2d_img(undersampled, axes=(-1, -2)), coil_dim=2).view(-1, 1, y, x)
        fully_sampled = root_sum_of_squares(ifft_2d_img(fully_sampled, axes=(-1, -2)), coil_dim=2).view(-1, 1, y, x)
        fs_estimate = self.model(aliased)
        
        loss =  torch.nn.functional.mse_loss(fs_estimate, fully_sampled)
        if batch_index == 0:
            gt_imgs = fully_sampled.view(b, t, y, x)
            gt_imgs = gt_imgs[0, :, :, :].unsqueeze(1).abs()
            grid = make_grid(gt_imgs, normalize=True, value_range=(0, gt_imgs.max()/4))
            self.logger.log_image("train/gt_images", [wandb.Image(grid, caption="Validation Ground Truth Images")])
            # imgs [b, t, h, w]
            es_imgs = fs_estimate.view(b, t, y, x)
            es_imgs = es_imgs[0, :, :, :].unsqueeze(1).abs()
            grid = make_grid(es_imgs, normalize=True, value_range=(0, gt_imgs.max()/4))
            self.logger.log_image("train/estimate_images", [wandb.Image(grid, caption="Validation Images")])
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_index: int): 
        undersampled, fully_sampled, _ = batch

        b, t, c, y, x = undersampled.shape
        aliased = root_sum_of_squares(ifft_2d_img(undersampled, axes=(-1, -2)), coil_dim=2).reshape(-1, 1, y, x)
        fully_sampled = root_sum_of_squares(ifft_2d_img(fully_sampled, axes=(-1, -2)), coil_dim=2).reshape(-1, 1, y, x)
        fs_estimate = self.model(aliased)
        loss =  torch.nn.functional.mse_loss(fs_estimate, fully_sampled)

        ssim = metrics.calculate_ssim(fully_sampled, fs_estimate, self.device)
        nmse = metrics.calculate_nmse(fully_sampled, fs_estimate)
        psnr = metrics.calculate_psnr(fully_sampled, fs_estimate, self.device)

        self.log_dict(
                {'val/loss': loss, 'val/ssim': ssim, 'val/psnr': psnr, 'val/nmse': nmse},
                on_epoch=True, prog_bar=True, logger=True, sync_dist=True
                )
        if batch_index == 0: 
            gt_imgs = fully_sampled.view(b, t, y, x)
            gt_imgs = gt_imgs[0, :, :, :].unsqueeze(1).abs()
            grid = make_grid(gt_imgs, normalize=True, value_range=(0, gt_imgs.max()/4))
            self.logger.log_image("val/gt_images", [wandb.Image(grid, caption="Validation Ground Truth Images")])
            # imgs [b, t, h, w]
            es_imgs = fs_estimate.view(b, t, y, x)
            es_imgs = es_imgs[0, :, :, :].unsqueeze(1).abs()
            grid = make_grid(es_imgs, normalize=True, value_range=(0, gt_imgs.max()/4))
            self.logger.log_image("val/estimate_images", [wandb.Image(grid, caption="Validation Images")])


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
                on_epoch=True, prog_bar=True, logger=True, sync_dist=True
                )
        return metrics_dict


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
            self, 
            in_chan, 
            out_chan, 
            depth=4,
            chans=18, 
            drop_prob=0
            ):
        super().__init__()

        cur_chan = chans

        # populate module list of number of down sampling layers
        self.down_sample_layers = nn.ModuleList([double_conv(in_chan, chans, drop_prob)])
        for _ in range(depth):
            self.down_sample_layers.append(Unet_down(cur_chan, cur_chan*2, drop_prob))
            cur_chan *= 2

        # populate module list of up sampling layers
        self.up_sample_layers = nn.ModuleList()
        for _ in range(depth):
            self.up_sample_layers.append(Unet_up(cur_chan, cur_chan//2, drop_prob))
            cur_chan //= 2

        # final convolution at the end 
        self.conv2d = nn.Conv2d(chans, out_chan, 1, bias=False)

    def forward(self, x):
        x, pad_sizes = self.pad(x)

        # save output of each down sampling layer
        stack = []

        # down sample layers
        for layer in self.down_sample_layers:
            x = layer(x)
            stack.append(x)

        # concatenate and pass through upsampling layers
        for i, layer in enumerate(self.up_sample_layers):
            x = layer(x, stack[-i - 2])

        x = self.conv2d(x)
        x = self.unpad(x, *pad_sizes)
        return x

    # pad input image to be divisible by 16 for unet downsampling
    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    # unpad unet input
    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]



class Unet_down(nn.Module):
    def __init__(self, in_channel, out_channel, drop_prob):
        super().__init__()
        self.down = down()
        self.conv = double_conv(in_channel, out_channel, drop_prob)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x

class Unet_up(nn.Module):
    def __init__(self, in_chan, out_chan, drop_prob):
        super().__init__()
        self.up = up(in_chan, out_chan)
        self.concat = concat()
        self.conv = double_conv(in_chan, out_chan, drop_prob)

    def forward(self, x, x_concat):
        x = self.up(x)
        x = self.concat(x, x_concat)
        x = self.conv(x)
        return x


class double_conv(nn.Module):
    def __init__(self, in_chans, out_chans, drop_prob):
        
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )
      
    def forward(self, x):
        return self.layers(x)

class down(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.AvgPool2d(2, stride=(2, 2))

    def forward(self, x):
        x = self.max_pool(x)
        return x


class up(nn.Module):

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.layers = nn.Sequential(
          nn.ConvTranspose2d(in_chan, out_chan, stride=2, kernel_size=2, bias=False),
          nn.InstanceNorm2d(out_chan, affine=True),
          nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class concat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, x_concat: torch.Tensor):
        x_concat_shape = x_concat.shape[-2:]
        x_shape = x.shape[-2:]
        diff_x = x_concat_shape[0] - x_shape[0]
        diff_y = x_concat_shape[1] - x_shape[1]
        x_concat_trimmed = x_concat
        if diff_x != 0:
            print('Different sizes! Trimming x')
            x_concat_trimmed = x_concat_trimmed[:, :, diff_x//2:-diff_x//2, :]
        if diff_y != 0:
            print('Different sizes! Trimming y')
            x_concat_trimmed = x_concat_trimmed[:, :, :, diff_y//2:-diff_y//2]
        concated_data = torch.cat((x, x_concat_trimmed), dim=1)
        return concated_data
