import torch 
import einops
from torch.fft import ifftshift, fftshift, ifft2, fft2

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

def root_sum_of_squares(data: torch.Tensor, coil_dim=0):
    return torch.sqrt(data.abs().pow(2).sum(coil_dim) + 1e-6)

fft_2d_img = lambda x, axes=[-1, -2]: fftshift(ifft2(ifftshift(x, dim=axes), dim=axes), dim=axes)
ifft_2d_img = lambda x, axes=[-1, -2]: ifftshift(fft2(fftshift(x, dim=axes), dim=axes), dim=axes)
