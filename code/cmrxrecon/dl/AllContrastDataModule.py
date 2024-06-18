from collections.abc import Callable
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
                transforms=NormalizeKSpace()
                )

        self.all_contrast_train, self.all_contrast_val = random_split(
            all_contrast_full, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.all_contrast_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.all_contrast_val, batch_size=self.batch_size, num_workers=self.num_workers)


class NormalizeKSpace(object):
    """Normalize k space to 1 for each slice 
    """

    def __call__(self, sample):
        # dimensions [t, h, w]
        under, fully_sampled = sample

        return under/under.abs().amax((-1, -2), keepdim=True), fully_sampled/under.abs().amax((-1, -2), keepdim=True)


