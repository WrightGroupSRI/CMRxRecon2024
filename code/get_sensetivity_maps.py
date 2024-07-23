import h5py 
import os
import os
import fnmatch

from cmrxrecon.espirit import espirit 
from cmrxrecon.dl.lowrank_varnet import ifft_2d_img
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend

            

def find_h5_files(root_dir, ignore_dirs):
    h5_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Remove directories to ignore from the search
        found = 0
        for ignore in ignore_dirs: 
            if ignore in dirpath: 
                found = 1
        if found: 
            continue

        for filename in fnmatch.filter(filenames, '*.h5'):
            h5_files.append(os.path.join(dirpath, filename))
    return h5_files

# Example usage
path = '/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/'

directories_to_ignore = ['Mask_Task1', 'Mask_Task2', 'ImgSnaphot', 'UnderSample_Task1']
h5_files = find_h5_files(path, directories_to_ignore)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
for file in h5_files:
    
    with h5py.File(file) as fr: 
        print(file)
        if 'sensetivites' in file.lower(): 
            continue
        with torch.no_grad():
            if 'validation' in file.lower():
                key = 'kus'
            else:
                key = 'kspace_full'
                
            k_space = fr[key][:]
            k_space = k_space['real'] + 1j* k_space['imag']
            k_space = torch.from_numpy(k_space)
            maps = []
            for split in torch.split(k_space, 1, dim=0):
                map = espirit(split[:, 0, ...].permute(0, 2, 3, 1).to(device), 4, 16, 0.001, 0.99, device)
                maps.append(map.permute(0, 3, 1, 2))

            maps = torch.concat(maps)
            print(maps.shape)
            
            dirname = os.path.dirname(file)
            basename = os.path.basename(file)
            patient_name = os.path.splitext(file)[0]

            sense_map_name = patient_name + '_sensetivites.h5'

            with h5py.File(os.path.join(dirname, sense_map_name), 'w') as fr: 
                print(fr.filename)
                fr.create_dataset('sensetivity', data=maps.cpu().numpy())


