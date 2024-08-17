import os
import numpy as np
import h5py 

from typing import Callable, Literal, Optional, Tuple, List
import torch
from torch.utils.data import Dataset, Subset
from dataclasses import dataclass
from cmrxrecon.espirit import espirit
from cmrxrecon.dl.dataloaders.VolumeDataset import VolumeDataset


class SliceDataset(Dataset):
    """Volume dataset for MICCAI 2024 challenge. 
    """

    def __init__(
            self, 
            dataset,
            transforms: Callable
            ):
        """
        Decroator for mri dataset to convert it to slice by slice

        Converts a volume dataset to a slice dataset

        Examples:
            # Description of my example.
            dataset = SliceDataset(VolumeDataset)
        """

        super().__init__()

        self.transforms = transforms
        self.volume_dataset = dataset
        
        self.slices = []
        if isinstance(dataset, Subset): 
            self.volume_index_lookup_table = []
            for subset_index in dataset.indices:
                self.slices.append(self.volume_dataset.dataset.file_list[subset_index].slices)
                self.volume_index_lookup_table.append(subset_index)
        else:
            for file in self.volume_dataset.file_list:
                self.slices.append(file.slices)

        
    def __len__(self):
        return sum(self.slices)


    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vol_idx, slice_idx = self.get_vol_slice_index(index)
        if isinstance(self.volume_dataset, Subset): 
            vol_idx = self.volume_index_lookup_table[vol_idx]
            dataset = self.volume_dataset.dataset
        else: 
            dataset = self.volume_dataset
        fully_sampled, under_sampled, sense = dataset[vol_idx]
        
        training_sample = (fully_sampled[slice_idx], under_sampled[slice_idx], sense[slice_idx], {})
        
        if self.transforms: 
            training_sample = self.transforms(training_sample)

        return training_sample 


    def get_vol_slice_index(self, index) -> Tuple[int, int]:
        slices = self.slices
        cumulative_sum = np.cumsum(slices)
        volume_index = np.where(cumulative_sum > index)[0][0]
        # get the slice index
        if volume_index > 0:
            slice_index = index - cumulative_sum[volume_index - 1] 
        else: 
            slice_index = index
        return volume_index, slice_index



import matplotlib.pyplot as plt
if __name__ == '__main__':
    path = '/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/Cine/'
    dataset = VolumeDataset(path, True, False, '4', 'cine_sax')
    x = dataset[0]
    print(x[0].shape)
    #fig, ax = plt.subplots(2, 1)
    #ax[0].imshow(x[0][0, 0, :, :].abs(), vmax=x[0].abs().max()/100)
    #ax[1].imshow(x[1][0, 0, :, :].abs(), vmax=x[0].abs().max()/100)
    #plt.show()
    assert torch.all(x[0] == x[1])

