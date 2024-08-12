import torch 
import einops
from torch.fft import ifftshift, fftshift, ifft2, fft2

def view_as_real(data): 
    shape = data.shape
    real_data = torch.view_as_real(data)
    real_data = real_data.contiguous().reshape(shape[0], shape[1]*2, *shape[2:])
    return real_data

def view_as_complex(data):
    shape = data.shape
    data = data.contiguous().reshape(shape[0], shape[1]//2, *shape[2:], 2)
    complex_data = torch.view_as_complex(data)
    return complex_data

def root_sum_of_squares(data: torch.Tensor, coil_dim=0):
    return torch.sqrt(data.abs().pow(2).sum(coil_dim) + 1e-6)


def pad_to_shape(tensor, target_shape):
    k_space_shape = tensor.shape
    pad_x = (target_shape[1] - k_space_shape[-1]) // 2
    pad_y = (target_shape[0] - k_space_shape[-2]) // 2
    padding = (pad_x, pad_x, pad_y, pad_y)  # (left, right, top, bottom)
    original_size = (k_space_shape[-1], k_space_shape[-2])
    padded_tensor = torch.nn.functional.pad(tensor, padding, "constant", 0)
    return padded_tensor, original_size

def crop_to_shape(tensor, original_size):
    tensor_shape = tensor.shape
    diff_x = (tensor_shape[-1] - original_size[0])//2
    diff_y = (tensor_shape[-2] - original_size[1])//2
    return tensor[..., diff_y:original_size[1] + diff_y, diff_x:original_size[0] + diff_x]

def fft_2d_img(x, axes=[-1, -2]): 
    return fftshift(fft2(ifftshift(x, dim=axes), dim=axes, norm='ortho'), dim=axes)
def ifft_2d_img(x, axes=[-1, -2]): 
    return ifftshift(ifft2(fftshift(x, dim=axes), dim=axes, norm='ortho'), dim=axes)
