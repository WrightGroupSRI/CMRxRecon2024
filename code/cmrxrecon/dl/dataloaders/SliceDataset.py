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
        self.volume_datset = dataset
        
        self.slices = []
        if isinstance(dataset, Subset): 
            self.volume_index_lookup_table = []
            for subset_index in dataset.indices:
                self.slices.append(self.volume_datset.dataset.file_list[subset_index].slices)
                self.volume_index_lookup_table.append(subset_index)


    def __len__(self):
        return sum(self.slices)


    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        vol_idx, slice_idx = self.get_vol_slice_index(index)
        if isinstance(self.volume_datset, Subset): 
            vol_idx = self.volume_index_lookup_table[vol_idx]
            file_list = self.volume_datset.dataset.file_list
        else:
            file_list = self.volume_datset.file_list
        
        subject_files = file_list[vol_idx]

        fs_file = subject_files.fully_sampled
        sense_file = subject_files.sensetivities
        try:
            index = torch.randint(len(subject_files.mask), size=(1,))
            validation = False
        except:
            index = 0
            validation = True

        mask_file = subject_files.mask[index]

        try:
            with h5py.File(fs_file, 'r') as fr:
                # DATA SHAPE [z, t, c, y, x]
                k_space_key = 'kspace_full'

                k_space:np.ndarray = fr[k_space_key][slice_idx]
                k_space = torch.from_numpy(k_space['real'] + 1j * k_space['imag'])

            if not validation:
                with h5py.File(mask_file, 'r') as fr:
                    # DATA SHAPE [z, t, c, y, x]
                    mask = torch.as_tensor(fr['mask'][:])

                with h5py.File(sense_file, 'r') as fr: 
                    # DATA SHAPE [z, c, y, x]
                    sensetivity = torch.from_numpy(fr['sensetivity'][slice_idx])
                    sensetivity = sensetivity[:]
                    sensetivity[0] = sensetivity[0].abs()
            else:
                mask = k_space != 0
                mask = mask[:, 0, :, :]
                sensetivity = np.ones_like(k_space)
                sensetivity = sensetivity[0, :, :, :]
        except Exception as e:
            print(f"ERROR")
            print(e)
            print(f"couldn't find one of these files! {fs_file} {mask_file} {sense_file}")
            raise e
        
        assert k_space != None
        assert mask != None
        assert sensetivity != None

        assert isinstance(k_space, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(sensetivity, torch.Tensor)
        
        # data shape [z, c, y , x]
        mask = mask.bool()

        sensetivity = sensetivity.unsqueeze(0)
        mask = mask.unsqueeze(1)
        training_sample = (k_space*mask, k_space, sensetivity)
        
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

