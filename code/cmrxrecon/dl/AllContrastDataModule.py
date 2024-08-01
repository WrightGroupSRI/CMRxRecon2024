import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torch 
from cmrxrecon.dl.dataloaders.AllContrastDataset import AllContrastDataset
from torchvision.transforms import Compose
from cmrxrecon.utils import fft_2d_img, ifft_2d_img



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
                transforms=Compose([NormalizeKSpace(), ZeroPadKSpace()]),
                task_one=True,
                file_extension=self.file_extension
                )

        self.all_contrast_train, self.all_contrast_val, self.all_contrast_test = random_split(
            all_contrast_full, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
        )



    def train_dataloader(self):
        return DataLoader(
                self.all_contrast_train, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                pin_memory=True,
                shuffle=True,
                collate_fn=collate_fn
                )

    def val_dataloader(self):
        return DataLoader(
                self.all_contrast_val, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                pin_memory=True,
                collate_fn=collate_fn, 
                shuffle=False
                )

    def test_dataloader(self):
        return DataLoader(
                self.all_contrast_test, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                pin_memory=True,
                collate_fn=collate_fn
                )


def pad_to_max_time(data:torch.Tensor, max_time:int):
    return torch.nn.functional.pad(data, (0, 0, 0, 0, 0, 0, 0, max_time - data.shape[0]))

def collate_fn(batch):
    max_time = max([x.shape[0] for x, _, _ in batch])
    us = [pad_to_max_time(x, max_time) for x, _, _ in batch]
    fs = [pad_to_max_time(x, max_time) for _, x, _ in batch]
    return (torch.stack(us), torch.stack(fs), torch.stack([sense for _, _, sense in batch]))

class NormalizeKSpace(object):
    """Normalize k space to 1 for each slice 
    """

    def __call__(self, sample:torch.Tensor):
        # dimensions [t, h, w]
        under, fully_sampled, sense = sample
        return under/under.abs().max(), fully_sampled/under.abs().max(), sense



class ZeroPadKSpace(object):
    """Zero pad k-space data to [256, 512] in [x, y] dimensions.
    """

    def __call__(self, sample):
        under, fully_sampled, sense = sample
        under = self.pad_to_shape(under, [256, 512])
        fully_sampled = self.pad_to_shape(fully_sampled, [256, 512])
        sense = ifft_2d_img(self.pad_to_shape(fft_2d_img(sense), [256, 512]))
        sense_mag = torch.sqrt((sense * sense.conj()).sum(dim=1, keepdim=True))
        return under, fully_sampled, sense/sense_mag

    def pad_to_shape(self, tensor, target_shape):
        _, _, x, y = tensor.shape
        pad_x = (target_shape[0] - x) // 2
        pad_y = (target_shape[1] - y) // 2
        padding = (pad_y, pad_y, pad_x, pad_x)  # (left, right, top, bottom)
        return torch.nn.functional.pad(tensor, padding, "constant", 0)


