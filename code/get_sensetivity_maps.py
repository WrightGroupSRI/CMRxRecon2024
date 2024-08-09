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
import argparse
from torchvision.utils import make_grid
import multiprocessing
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

def save_files(file): 
    try:
        with h5py.File(file) as fr: 
            print(file)
            if 'sensetivites' in file.lower(): 
                return
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
                map = espirit(split[:, 0, ...].permute(0, 2, 3, 1).to(device), 5, 16, 0.0001, 0.99, device)
                maps.append(map.permute(0, 3, 1, 2))
 
            maps = torch.concat(maps, dim=0)
            print(maps.shape)
            dirname = os.path.dirname(file)
            basename = os.path.basename(file)
            patient_name = os.path.splitext(file)[0]

            sense_map_name = patient_name + '_sensetivites.h5'

            with h5py.File(os.path.join(dirname, sense_map_name), 'w') as fr: 
                print(f'Saving to {fr.filename}')
                fr.create_dataset('sensetivity', data=maps.cpu().numpy())
    except OSError as e:
        # Print the error message and the file name
        print(f"OS error: {e}")
        print(f"Error occurred in file: {file}")

# Example usage
parser = argparse.ArgumentParser()
default_path = '/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/Tagging//' 
parser.add_argument('--path', type=str, default=default_path)
args = parser.parse_args()
path = args.path

directories_to_ignore = ['Mask_Task1', 'Mask_Task2', 'ImgSnaphot', 'UnderSample_Task1']
h5_files = find_h5_files(path, directories_to_ignore)
#h5_files = ['/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/Cine/TrainingSet/FullSample/P106/cine_lvot.h5']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

cpus_per_task = int(os.getenv('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
print(cpus_per_task)
pool = multiprocessing.Pool(processes=cpus_per_task)

# Use pool.map to apply the process_file function to each input file
pool.map(save_files, h5_files)

# Close the pool and wait for all worker processes to finish
pool.close()
pool.join()
