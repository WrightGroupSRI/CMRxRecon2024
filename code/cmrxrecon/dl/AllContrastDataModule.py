import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torch 
from cmrxrecon.dl.dataloaders.AllContrastDataset import AllContrastDataset



class AllContrastDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 1, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        all_contrast_full = AllContrastDataset(
                self.data_dir, 
                train=True,
                transforms=NormalizeKSpace(),
                task_one=False
                )

        self.all_contrast_train, self.all_contrast_val = random_split(
            all_contrast_full, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
                self.all_contrast_train, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                pin_memory=True,
                shuffle=True
                )

    def val_dataloader(self):
        return DataLoader(
                self.all_contrast_val, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                pin_memory=True
                )

class NormalizeKSpace(object):
    """Normalize k space to 1 for each slice 
    """

    def __call__(self, sample):
        # dimensions [t, h, w]
        under, fully_sampled = sample
        return under/under.abs().amax((-1, -2), keepdim=True), fully_sampled/under.abs().amax((-1, -2), keepdim=True)


class ZeroPadKSpace(object):
    """Zero pad k-space data to [256, 512] in [x, y] dimensions.
    """

    def __call__(self, sample):
        under, fully_sampled = sample
        under = self.pad_to_shape(under, [256, 512])
        fully_sampled = self.pad_to_shape(fully_sampled, [256, 512])
        return under, fully_sampled

    def pad_to_shape(self, tensor, target_shape):
        _, _, x, y = tensor.shape
        pad_x = (target_shape[0] - x) // 2
        pad_y = (target_shape[1] - y) // 2
        padding = (pad_y, pad_y, pad_x, pad_x)  # (left, right, top, bottom)
        return torch.nn.functional.pad(tensor, padding, "constant", 0)
