from cmrxrecon.dl.dataloaders.MRIDataset import MRIDataset
from torch.utils.data import Dataset
import os
import numpy as np
from typing import Literal, Tuple, Optional, Callable
from dataclasses import dataclass
import torch

class AllContrastDataset(Dataset):
    """Dataloader that loads all contrasts into on dataset"""
    def __init__(
            self, 
            parent_dir: str,
            task_one: bool = True,
            train: bool = True,
            acceleration_factor: Literal['4', '8', '10'] = '4',
            transforms: Optional[Callable] = None,
            file_extension: str = ".h5"
            ):
        """
        Initalize Dataset with input parameters with all contrats

        Args:
            parent_dir (str): Path to top level directory containing all contrasts
            task_one (bool): Flag to set if the dataloader should load task one or two
            train (bool): Flag to set if the dataloader is training or validation
            acceleration_factor (Literal['4', '8', '10']): Acceleration factor for dataset

        Returns:
            type and description of the returned object.

        """
        
        contrasts_names = os.listdir(parent_dir)
        self.datasets = []
        self.transforms = transforms
        for contrast in contrasts_names: 

            path = os.path.join(parent_dir, contrast)

            match contrast.lower():
                case 'aorta':
                    prefixes = ['aorta_sag', 'aorta_tra']
                case 'cine':
                    prefixes = ['cine_lvot', 'cine_lax', 'cine_sax']
                case 'tagging':
                    prefixes = ['tagging']
                case 'mapping':
                    prefixes = ['T1map', 'T2map']
                case 'blackblood':
                    if train:
                        continue
                    prefixes = ['blackblood']
                case 'flow2d': 
                    if train:
                        continue
                    prefixes = ['flow2d']
                case _:
                    print(f'Validation dataset found! {contrast.lower}')
                    continue

            for prefix in prefixes:
                self.datasets.append(
                       MRIDataset(
                           path, 
                           task_one=task_one, 
                           train=train,
                           acceleration_factor=acceleration_factor, 
                           file_prefix = prefix, 
                           transforms=transforms, 
                           file_extension=file_extension
                           )
                       )
        self.dataset_lengths = [len(dataset) for dataset in self.datasets]

    def __len__(self):
        return sum(self.dataset_lengths)

    def get_dataset_index_and_index(self, index) -> Tuple[int, int]:
        cumulative_sum = np.cumsum(self.dataset_lengths)
        dataset_index = np.where(cumulative_sum > index)[0][0]
        # get the slice index
        if dataset_index> 0:
            index = index - cumulative_sum[dataset_index - 1] 
        else: 
            index = index
        return dataset_index, index

    def __getitem__(self, index): 
        dataset_index, idx = self.get_dataset_index_and_index(index)
        return self.datasets[dataset_index][idx]


@dataclass
class Shape:
    x: int
    y: int
    coils: int
    t: int

if __name__ == '__main__':
    dataset = AllContrastDataset(
            parent_dir='/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/',
            acceleration_factor='4', 
            task_one=True,
            train=True,
            )
    unique_shapes = []

    all_shapes = torch.ones(len(dataset), 4)

    for i, data in enumerate(dataset):
        
        all_shapes[i, :] = torch.tensor([data[0].shape[0], data[0].shape[1], data[0].shape[2], data[0].shape[3]])
        if i > 50:
            break

    print(all_shapes.shape)
    unique_rows = torch.unique(all_shapes, dim=0)
    print(unique_rows)

