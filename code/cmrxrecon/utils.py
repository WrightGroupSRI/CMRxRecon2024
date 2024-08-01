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


def pad_to_shape(tensor, target_shape):
    _, _, x, y = tensor.shape
    pad_x = (target_shape[0] - x) // 2
    pad_y = (target_shape[1] - y) // 2
    padding = (pad_y, pad_y, pad_x, pad_x)  # (left, right, top, bottom)
    original_size = (x, y)
    padded_tensor = torch.nn.functional.pad(tensor, padding, "constant", 0)
    return padded_tensor, original_size

def crop_to_shape(tensor, original_size):
    _, _, x, y = tensor.shape
    diff_x = (original_size[0] - x)//2
    diff_y = (original_size[1] - x)//2
    return tensor[:, :, diff_x:original_size[0], diff_y:original_size[1]]

fft_2d_img = lambda x, axes=[-1, -2]: fftshift(ifft2(ifftshift(x, dim=axes), dim=axes, norm='ortho'), dim=axes)
ifft_2d_img = lambda x, axes=[-1, -2]: ifftshift(fft2(fftshift(x, dim=axes), dim=axes, norm='ortho'), dim=axes)
