import math
import pytorch_lightning as pl

import torch 
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose
import torch.nn.functional as F

from cmrxrecon.dl.dataloaders.AllContrastDataset import AllContrastDataset
from cmrxrecon.dl.dataloaders.SliceDataset import SliceDataset


class AllContrastDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 1, num_workers: int = 0, file_extension=".h5"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.file_extension = file_extension

    def setup(self, stage):
        all_contrast_full = AllContrastDataset(
                self.data_dir, 
                train=True,
                task_one=False,
                file_extension=self.file_extension
                )

        self.all_contrast_train, self.all_contrast_val = random_split(
            all_contrast_full, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
        )
        print(f'Train dataset has {len(self.all_contrast_train)} volumes')
        print(f'Val dataset has {len(self.all_contrast_val)} volumes')

        self.all_contrast_train_slices = SliceDataset(self.all_contrast_train, transforms=Compose([NormalizeKSpace(), ZeroPadKSpace()]))
        self.all_contrast_val_slices = SliceDataset(self.all_contrast_val, transforms=Compose([NormalizeKSpace(), ZeroPadKSpace()]))

        print(f'Train slice dataset has {len(self.all_contrast_train_slices)} slices')
        print(f'Val slice dataset has {len(self.all_contrast_val_slices)} slices')



    def train_dataloader(self):
        return DataLoader(
                self.all_contrast_train_slices, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                pin_memory=True,
                shuffle=True,
                collate_fn=collate_fn
                )

    def val_dataloader(self):
        return DataLoader(
                self.all_contrast_val_slices, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                pin_memory=True,
                collate_fn=collate_fn, 
                shuffle=False
                )

"""
Pads k-t k-space so time dimension is all the same. Padded time dimension with zeros
"""
def pad_to_max_time(data:torch.Tensor, max_time:int):
    return torch.nn.functional.pad(data, (0, 0, 0, 0, 0, 0, 0, max_time - data.shape[0]))

def collate_fn(batch):
    max_time = max([x.shape[0] for x, _, _, _ in batch])
    us = [pad_to_max_time(x, max_time) for x, _, _, _ in batch]
    fs = [pad_to_max_time(x, max_time) for _, x, _, _ in batch]
    transforms = [(x) for _, _, _, x in batch]
    return (torch.stack(us), torch.stack(fs), torch.stack([sense for _, _, sense, _ in batch]),  transforms)

class NormalizeKSpace(object):
    """Normalize k space to 1 for each slice 
    """

    def __call__(self, sample):
        # dimensions [t, h, w]
        under, fully_sampled, sense, transform_params = sample
        scaling_factor = under.abs().max()
        transform_params['scaling_factor'] = scaling_factor
        return under/scaling_factor, fully_sampled/scaling_factor, sense, transform_params

class UnNormKSpace(object):
    def __call__(self, output, scaling):
        return output * scaling


class ZeroPadKSpace(object):
    """Zero pad k-space data to [256, 512] in [x, y] dimensions.
    """

    def __call__(self, sample):
        under, fully_sampled, sense, transform_params = sample
        t, c, h, w = under.shape
        transform_params['original_size'] = (h, w)
        under = self.pad_to_shape(under, [256, 512])
        fully_sampled = self.pad_to_shape(fully_sampled, [256, 512])
        sense = self.linear_interpolator(sense)
        mask = sense[0, 0, :, :] != 0
        scaling = (sense[:, :, mask].conj() * sense[:, :, mask]).sum(1)
        sense[:, :, mask] = sense[:, :, mask]/torch.sqrt(scaling)
        return under, fully_sampled, sense, transform_params

    def pad_to_shape(self, tensor, target_shape):
        _, _, x, y = tensor.shape
        pad_x = (target_shape[0] - x) 
        pad_y = (target_shape[1] - y) 
        padding = (pad_y//2, math.ceil(pad_y/2), pad_x//2, math.ceil(pad_x/2))  # (left, right, top, bottom)
        return torch.nn.functional.pad(tensor, padding, "constant", 0)

    def linear_interpolator(self, tensor, target_shape=(256, 512)):

        # Perform interpolation
        interpolated_tensor_mag = F.interpolate(tensor.abs(), size=target_shape, mode='bilinear', align_corners=False)
        interpolated_tensor_angle = F.interpolate(torch.angle(tensor), size=target_shape, mode='nearest')


        return interpolated_tensor_mag * torch.exp(1j* interpolated_tensor_angle)

class UnpadKSPace(object):
    def __call__(self, output, original_shape):
        y = output.shape[-2]
        x = output.shape[-1]
        diff_y = y - original_shape[0]  
        diff_x = x - original_shape[1]  
        return output[..., diff_y//2:diff_y//2+original_shape[0], diff_x//2:diff_x//2+original_shape[1]]

        
        

