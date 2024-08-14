#!/usr/bin/env python3

###############################################################
# MAIN FUNCTION
# code that gets run in the docker container
#
# Calder Sheagren
# University of Toronto
# calder.sheagren@sri.utoronto.ca
# Date: June 26, 2023 
###############################################################

import os
import argparse
from datetime import datetime
import time
import numpy as np
import torch

import h5py 
import os
import os
import fnmatch

from get_sensetivity_maps import find_h5_files
from cmrxrecon.espirit_recon import espirit_recon
import torch
import matplotlib
import argparse
import multiprocessing

matplotlib.use('Agg')  # Use the 'Agg' backend
            


def calc_espirit(file): 
    try:
        with h5py.File(file) as fr: 
            print(file)
            if 'sensetivites' in file.lower(): 
                return
            if 'basis' in file.lower():
                return 
            if 'validation' in file.lower():
                key = 'kus'
            else:
                key = 'kspace_full'
                
            k_space = fr[key][:]
            k_space = k_space['real'] + 1j* k_space['imag']
            k_space = torch.from_numpy(k_space)
            #probs kz, kt, kc, kx, ky
            k_space = k_space.permute((1, 0, 2, 3, 4))

        with torch.no_grad():
            recon = espirit_recon(k_space)
 
            # probs t z x y 
            print(recon.shape)
            recon = torch.from_numpy(recon)
            recon = recon.reshape(1, 0, 2, 3)
            z, x, y, t = recon.shape
            recon = recon.reshape(z*x*y, t)
            temporal_basis, sv, spatial_basis = torch.linalg.svd(recon.view(z, t, x*y))
            spatial_basis = spatial_basis * sv.unqueeze(-1)
            spatial_basis = spatial_basis.reshape(z, t, x, y)

            dirname = os.path.dirname(file)
            patient_name = os.path.splitext(file)[0]

            basis_file_name = patient_name + '_bases.h5'

            with h5py.File(os.path.join(dirname, basis_file_name), 'w') as fr: 
                print(f'Saving to {fr.filename}')
                fr.create_dataset('spatial', data=spatial_basis.cpu().numpy())
                fr.create_dataset('temporal', data=spatial_basis.cpu().numpy())
    except OSError as e:
        # Print the error message and the file name
        print(f"OS error: {e}")
        print(f"Error occurred in file: {file}")

if name == '__main__':
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
    pool.map(calc_espirit, h5_files)

# Close the pool and wait for all worker processes to finish
    pool.close()
    pool.join()
