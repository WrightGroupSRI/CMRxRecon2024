#!/usr/bin/env python3

###############################################################
# LOWRANK RECONSTRUCTION
#
# Brenden Kadota
# University of Toronto
# brenden.kadota@gmail.com
# Date: April 27, 2024
###############################################################


import numpy as np
from cmrxrecon.dl.basis_denoiser_spatial import SpatialDenoiser
from cmrxrecon.dl.basis_denoiser_temporal import TemporalDenoiser
from cmrxrecon.dl.lowrank_solver import LowRankSolver
import torch
from cmrxrecon.espirit import espirit
from cmrxrecon.utils import ifft_2d_img, root_sum_of_squares
from torchvision.transforms import Compose
from cmrxrecon.dl.AllContrastDataModule import NormalizeKSpace, ZeroPadKSpace

SPATIAL_DENOISER_PATH = '/home/kadotab/scratch/cmrxrecon_checkpoints/2024-08-07_09_spatial_epoch=32-val/loss=0.78-val/ssim=0.67.ckpt'
TEMPORAL_DENOISER_PATH = '/home/kadotab/scratch/cmrxrecon_checkpoints/2024-08-07_11_temporal_epoch=29-val/loss=0.15-val/ssim=0.00.ckpt'


def calc_espirit(bart_kspace, device):
    maps = []
    for i in range(bart_kspace.shape[0]):
        map = (espirit(bart_kspace[[i], 0, ...].permute(0, 2, 3, 1).to(device), 5, 16, 0.0001, 0.99, device))
        maps.append(map.permute(0, 3, 1, 2))
    maps = torch.cat(maps, 0)
    return maps

def lowrank(kspace: np.ndarray, mask=None, device='cpu', lambda_reg=1e-1):
    ######## NOT SURE THE MASK DIMENSIONS BUT ASSUMING [sht, shy, shx]

    [sht, shz, shc, shy, shx] = kspace.shape

    mask = kspace != 0
    try:
        spatial_denoiser = SpatialDenoiser.load_from_checkpoint(checkpoint_path=SPATIAL_DENOISER_PATH)
        temporal_denoiser = TemporalDenoiser.load_from_checkpoint(checkpoint_path=TEMPORAL_DENOISER_PATH)
    except:
        spatial_denoiser = SpatialDenoiser()
        temporal_denoiser = TemporalDenoiser()

    solver = LowRankSolver(spatial_denoiser=spatial_denoiser, temporal_denoiser=temporal_denoiser, lambda_reg=1e-1, cascades=5)
    transforms = Compose([NormalizeKSpace(), ZeroPadKSpace()])

    solver.to(device)
    solver.eval()
    kspace = torch.from_numpy(kspace).to(device)

    # kspace now z, t, c, h, w
    kspace = torch.permute(kspace, (1, 0, 2, 3, 4))
    
    # UPDATE FOR YOUR ESPIRIT CODE 
    maps = calc_espirit(kspace, device)
    maps = maps.unsqueeze(1)
    #maps = torch.load('maps.pt')

    print(maps.shape)
    print(kspace.shape)
    k_space = [] 
    mask = []
    padded_maps = []

    for i in range(kspace.shape[0]): 
        values = transforms((kspace[i], kspace[i], maps[i]))
        k_space.append(values[0])
        mask.append(values[1])
        padded_maps.append(values[2])

    k_space = torch.stack(k_space, dim=0).to(device)
    mask  = torch.stack(mask, dim=0).to(device)
    padded_maps = torch.stack(padded_maps, dim=0).to(device)
    print(k_space.shape) 
    # solve for k-space
    with torch.no_grad():
        recon_k_space = solver(k_space, k_space != 0, padded_maps)

    # z, t, y, x
    recon_images = root_sum_of_squares(ifft_2d_img(recon_k_space), 2)

    # t, z, y, x
    recon_images = recon_images.permute(1, 0, 2, 3)

    return recon_images.cpu().numpy()

import h5py
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from torchvision.utils import make_grid
if __name__ == '__main__':
    file = '/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/Aorta/ValidationSet/UnderSample_Task2/P001/aorta_sag_kus_ktGaussian24.h5'
    data = None
    with h5py.File(file, 'r') as fr: 
        data = fr['kus'][:]
    
    data = data['real'] + 1j * data['imag']

    data = np.transpose(data, (1, 0, 2, 3, 4))
    recon = lowrank(data, mask=None, device='cuda')
    plt.imshow(make_grid(torch.from_numpy(recon[:, [0], :, :]))[0])
    plt.savefig('lowrank')


