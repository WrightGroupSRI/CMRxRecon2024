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

class VolumeDataset(Dataset):
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
        self.file_prefix = file_prefix
        self.task_one = task_one
        target_directory = self.setup_paths(directory)
        
        # populate list of patient files
        self.file_list:List[PatientFile] = []
        print(f'Counting Slices in {file_prefix}')
        for file in self.target_files:

            # create paths to fully sampled and mask files
            if self.train:
                file_name =  file_prefix + self.file_extension
            else:
                file_name = file_prefix + '_kus_Uniform' + acceleration_factor + '.h5'
            fs_file = os.path.join(target_directory, file, file_name)

            if not os.path.exists(fs_file):
                continue
            sense_file = os.path.join(target_directory, file, file_prefix + '_sensetivites.h5') 
            
            if task_one:
                mask_file = file_prefix + '_mask_Uniform' + acceleration_factor + '.h5' 
                mask_files = [os.path.join(self.mask_dir, file, mask_file)]
            else:
                
                mask_files = os.listdir(os.path.join(self.mask_dir, file))
                mask_files = [os.path.join(self.mask_dir, file, mask_file) for mask_file in mask_files if '.h5' in mask_file and file_prefix in mask_file]
            try:
                slices = None
                with h5py.File(fs_file, 'r') as fr:
                    # DATA SHAPE [t, z, c, y, x]
                    if self.train:
                        slices = fr['kspace_full'].shape[0]
                    else: 
                        slices = fr['kus'].shape[0]
                assert slices != None

            except Exception as e: 
                print(f'could not open {fs_file} with error {e}')
                raise

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
        print(f'Found {len(self.file_list)} Volumes!')

    def setup_paths(self, directory):
        if self.train:
            directory = os.path.join(directory, 'TrainingSet') # add trailing slash if not there
        else:
            directory = os.path.join(directory, 'ValidationSet') # add trailing slash if not there

        # Set up paths based on task and train/validation
        if self.train:
            target_direcory = os.path.join(directory, 'FullSample') 
        else:
            if self.task_one:
                target_direcory = os.path.join(directory,'UnderSample_Task1')
            else:
                target_direcory = os.path.join(directory,'UnderSample_Task2')

        # get all volume files from target directory
        self.target_files = os.listdir(target_direcory)
        self.target_files.sort()

        # select directory based on task
        if self.task_one:
            self.mask_dir = os.path.join(directory, 'Mask_Task1')
        else:
            self.mask_dir = os.path.join(directory, 'Mask_Task2')
        return target_direcory

    
    def __len__(self):
        return len(self.file_list)

    def get_random_mask_file(self, files:PatientFile):
        if self.train:
            index = torch.randint(len(files.mask), size=(1,))
            mask_file = files.mask[index]
        else:
            mask_file = ''
        return mask_file


    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        subject_files = self.file_list[index]

        fs_file = subject_files.fully_sampled
        sense_file = subject_files.sensetivities
        mask_file = self.get_random_mask_file(subject_files)

        k_space: torch.Tensor
        mask: torch.Tensor
        sensetivity: torch.Tensor
        # DATA SHAPE [z, t, c, y, x]
        try:
            with h5py.File(fs_file, 'r') as fr:
                if self.train:
                    key = 'kspace_full'
                else: 
                    key = 'kus'
                k_space = (fr[key][:])

            if self.train:
                with h5py.File(mask_file, 'r') as fr:
                    mask = torch.as_tensor(fr['mask'][:])

                with h5py.File(sense_file, 'r') as fr: 
                    sensetivity = torch.from_numpy(fr['sensetivity'][:])
                    # normalize the phase of the first mask to 0
                    sensetivity[:, 0] = sensetivity[:, 0].abs()

        except OSError as e: 
            print(f'os error: {e}')
            print(f"couldn't find one of these files! {fs_file} {mask_file} {sense_file}")
        except IndexError as e: 
            print(f'os error: {e}')
            print(f"couldn't find one of these files! {fs_file} {mask_file} {sense_file}")
        except Exception as e:
            print(f'error {e}')
            print(f"ERROR")
            print(f"couldn't find one of these files! {fs_file} {mask_file} {sense_file}")
            raise 

        
        # data shape [z, c, y , x]
        if self.train:
            mask = mask.bool()
        else:
            mask = torch.as_tensor(np.ones((k_space.shape[1], k_space.shape[2], k_space.shape[3]), dtype=int))
            sensetivity = torch.as_tensor(np.ones((k_space.shape[1], k_space.shape[2], k_space.shape[3]), dtype=int))

        if self.task_one:
            mask = mask.unsqueeze(0)

        k_space = torch.from_numpy(k_space['real'] + 1j * k_space['imag'])
        sensetivity = sensetivity.unsqueeze(1)

        mask = mask.unsqueeze(1)
        if self.train:
            training_sample = (k_space*mask, k_space, sensetivity)
        else:
            training_sample = (k_space, k_space, sensetivity)
        
        if self.transforms: 
            training_sample = self.transforms(training_sample)

        return training_sample 





import matplotlib.pyplot as plt
if __name__ == '__main__':
    path = '/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/Aorta/'
    dataset = VolumeDataset(path, False, True, '4', 'aorta_sag')
    x = dataset[0]
    print(x[0].shape)
    #fig, ax = plt.subplots(2, 1)
    #ax[0].imshow(x[0][0, 0, :, :].abs(), vmax=x[0].abs().max()/100)
    #ax[1].imshow(x[1][0, 0, :, :].abs(), vmax=x[0].abs().max()/100)
    #plt.show()
    assert torch.all(x[0] == x[1])

