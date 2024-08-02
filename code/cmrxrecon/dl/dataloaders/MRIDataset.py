import os
import numpy as np
import h5py 

from typing import Callable, Literal, Optional, Tuple, List
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from cmrxrecon.espirit import espirit


@dataclass
class PatientFile:
    fully_sampled: str
    mask: list[str]
    sensetivities: str
    slices: int

class MRIDataset(Dataset):
    """Volume dataset for MICCAI 2024 challenge. 
    """

    def __init__(
            self, 
            directory:str,
            task_one: bool = False, 
            train: bool = False, 
            acceleration_factor: Literal['4', '8', '10'] = '4',
            file_prefix: str = 'cine_lax',
            transforms: Optional[Callable] = None,
            all_data: bool = False, 
            file_extension: str = '.h5'
            ):
        """
        Initalize a contrast dataset based on parameters

        Args:
            directory (str): path to contrast directory head
            task_one (bool): flag for task one, else task two 
            train (bool): flag for training, else validation
            acceleration_factor (str): R value for acceleration factor
            file_prefix (str): contrast and slice axis field for file names
            transforms (callable): transforms to perform on output

        Returns:
            Returns a pytorch object

        """

        super().__init__()

        self.file_extension = file_extension
        self.transforms = transforms
        self.train = train
        
        if self.train:
            directory = os.path.join(directory, 'TrainingSet') # add trailing slash if not there
        else:
            directory = os.path.join(directory, 'ValidationSet') # add trailing slash if not there

        # Set up paths based on task and train/validation
        if self.train:
            target_direcory = os.path.join(directory, 'FullSample') 
        else:
            if task_one:
                target_direcory = os.path.join(directory,'UnderSample_Task1')
            else:
                target_direcory = os.path.join(directory,'UnderSample_Task2')

        # get all volume files from target directory
        self.target_files = os.listdir(target_direcory)

        # select directory based on task
        if task_one:
            self.mask_dir = os.path.join(directory, 'Mask_Task1')
        else:
            self.mask_dir = os.path.join(directory, 'Mask_Task2')

        # populate list of patient files
        self.file_list:List[PatientFile] = []
        print(f'Counting Slices in {file_prefix}')
        for file in self.target_files:

            # create paths to fully sampled and mask files
            if self.train:
                file_name =  file_prefix + self.file_extension
            else:
                file_name = file_prefix + '_kus_Uniform' + acceleration_factor + '.h5'
            fs_file = os.path.join(target_direcory, file, file_name)

            if not os.path.exists(fs_file):
                continue
            sense_file = os.path.join(target_direcory, file, file_prefix + '_sensetivites.h5') 
            
            if task_one:
                mask_file = file_prefix + '_mask_Uniform' + acceleration_factor + '.h5' 
                mask_files = [os.path.join(self.mask_dir, file, mask_file)]
            else:
                
                mask_files = os.listdir(os.path.join(self.mask_dir, file))
                mask_files = [os.path.join(self.mask_dir, file, mask_file) for mask_file in mask_files if '.mat' in mask_file and file_prefix in mask_file]

            with h5py.File(fs_file, 'r') as fr:
                # DATA SHAPE [t, z, c, y, x]
                if self.train:
                    slices = (fr['kspace_full'].shape[0])
                else: 
                    slices = fr['kus'].shape[0]
            if all_data:  
                for mask in mask_files: 
                    self.file_list.append(
                            PatientFile(fully_sampled=fs_file, mask=[mask], slices=slices, sensetivities=sense_file)
                            )
            else:
                self.file_list.append(
                PatientFile(fully_sampled=fs_file, mask=mask_files, slices=slices, sensetivities=sense_file)
                )
                
        print(f'Found {sum(patient.slices for patient in self.file_list)} slices!')
    
    def __len__(self):
        return sum(patient.slices for patient in self.file_list)


    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        vol_idx, slice_idx = self.get_vol_slice_index(index)
        
        subject_files = self.file_list[vol_idx]

        fs_file = subject_files.fully_sampled
        sense_file = subject_files.sensetivities
        try:
            index = torch.randint(len(subject_files.mask), size=(1,))
            mask_file = subject_files.mask[index]
            validation = False
        except:
            index = 0
            validation = True

        k_space: torch.Tensor
        mask: torch.Tensor
        sensetivity: torch.Tensor
        try:
            with h5py.File(fs_file, 'r') as fr:
                # DATA SHAPE [z, t, c, y, x]
                if self.train:
                    k_space = (fr['kspace_full'][slice_idx])
                else: 
                    k_space = (fr['kus'][slice_idx])

            if not validation:
                with h5py.File(mask_file, 'r') as fr:
                    # DATA SHAPE [z, t, c, y, x]
                    mask = torch.as_tensor(fr['mask'][:])

                with h5py.File(sense_file, 'r') as fr: 
                    # DATA SHAPE [z, c, y, x]
                    sensetivity = torch.from_numpy(fr['sensetivity'][:])
                    sensetivity = sensetivity[slice_idx]
        except OSError as e: 
            print(f'os error: e')
            self.create_new_espirit_map(fs_file, 'cuda')
            with h5py.File(sense_file, 'r') as fr: 
                # DATA SHAPE [z, c, y, x]
                sensetivity = torch.from_numpy(fr['sensetivity'][:])
                sensetivity = sensetivity[slice_idx]
        except IndexError as e: 
            print(f'os error: e')
            self.create_new_espirit_map(fs_file, 'cuda')
            with h5py.File(sense_file, 'r') as fr: 
                # DATA SHAPE [z, c, y, x]
                sensetivity = torch.from_numpy(fr['sensetivity'][:])
                sensetivity = sensetivity[slice_idx]

        #except:
        #    print(f"ERROR")
        #    print(f"couldn't find one of these files! {fs_file} {mask_file} {sense_file}")

        
        # data shape [z, c, y , x]
        if not validation:
            mask = mask.bool()
        else:
            mask = torch.as_tensor(np.ones((k_space.shape[1], k_space.shape[2], k_space.shape[3]), dtype=int))
            sensetivity = torch.as_tensor(np.ones((k_space.shape[1], k_space.shape[2], k_space.shape[3]), dtype=int))
        
        k_space = torch.from_numpy(k_space['real'] + 1j * k_space['imag'])
        if k_space.shape[0] == 1:
            print('only found one time dimension')
            k_space = k_space.repeat(3, 1, 1, 1)
        sensetivity = sensetivity.unsqueeze(0)

        #mask = mask.unsqueeze(1)
        if not validation:
            training_sample = (k_space*mask.unsqueeze(0).unsqueeze(0), k_space, sensetivity)
        else:
            training_sample = (k_space, k_space, sensetivity)
        
        if self.transforms: 
            training_sample = self.transforms(training_sample)
        return training_sample 


    def get_vol_slice_index(self, index) -> Tuple[int, int]:
        slices = [patient.slices for patient in self.file_list]
        cumulative_sum = np.cumsum(slices)
        volume_index = np.where(cumulative_sum > index)[0][0]
        # get the slice index
        if volume_index > 0:
            slice_index = index - cumulative_sum[volume_index - 1] 
        else: 
            slice_index = index
        return volume_index, slice_index


    def create_new_espirit_map(self, file, device):
        try:
            with h5py.File(file, 'r') as fr: 
                print(file)
                if 'validation' in file.lower():
                    key = 'kus'
                else:
                    key = 'kspace_full'
                    
                k_space = fr[key][:]
                k_space = k_space['real'] + 1j* k_space['imag']
                k_space = torch.from_numpy(k_space)

            with torch.no_grad():
                maps = []
                for split in torch.split(k_space, 1, dim=0):
                    map = espirit(split[:, 0, ...].permute(0, 2, 3, 1).to(device), 8, 16, 0.001, 0.99, device)
                    maps.append(map.permute(0, 3, 1, 2))

                maps = torch.concat(maps)
                print("sensetivity", maps.shape)
                
                dirname = os.path.dirname(file)
                basename = os.path.basename(file)
                patient_name = os.path.splitext(file)[0]

                sense_map_name = patient_name + '_sensetivites.h5'

                with h5py.File(os.path.join(dirname, sense_map_name), 'w') as fr: 
                    print(fr.filename)
                    fr.create_dataset('sensetivity', data=maps.cpu().numpy())
        except OSError as e:
            # Print the error message and the file name
            print(f"OS error: {e}")
            print(f"Error occurred in file: {file}")

import matplotlib.pyplot as plt
if __name__ == '__main__':
    path = '/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/Cine/'
    dataset = MRIDataset(path, True, False, '4', 'cine_sax')
    x = dataset[0]
    print(x[0].shape)
    #fig, ax = plt.subplots(2, 1)
    #ax[0].imshow(x[0][0, 0, :, :].abs(), vmax=x[0].abs().max()/100)
    #ax[1].imshow(x[1][0, 0, :, :].abs(), vmax=x[0].abs().max()/100)
    #plt.show()
    assert torch.all(x[0] == x[1])

