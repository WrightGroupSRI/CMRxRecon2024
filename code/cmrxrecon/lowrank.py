#!/usr/bin/env python3

###############################################################
# LOWRANK RECONSTRUCTION
#
# Brenden Kadota
# University of Toronto
# brenden.kadota@gmail.com
# Date: April 27, 2024
###############################################################

import pytorch_lightning

from .smaps import maps
import numpy as np
from cmrxrecon.dl.lowrank_varnet import LowRankModl
from cmrxrecon.dl.basis_denoiser_spatial import SpatialDenoiser
from cmrxrecon.dl.basis_denoiser_temporal import TemporalDenoiser
from cmrxrecon.dl.lowrank_solver import LowRankSolver
import torch
from cmrxrecon.espirit import espirit
from cmrxrecon.utils import ifft_2d_img, root_sum_of_squares

SPATIAL_DENOISER_PATH = '/home/kadotab/scratch/cmrxrecon_checkpoints/2024-07-24_09epoch=8-val/loss=0.00-val/ssim=0.00.ckpt'
TEMPORAL_DENOISER_PATH = '/home/kadotab/scratch/cmrxrecon_checkpoints/2024-07-24_09epoch=8-val/loss=0.00-val/ssim=0.00.ckpt'

def lowrank(kspace: np.ndarray, mask=None, device='cpu'):
    ######## NOT SURE THE MASK DIMENSIONS BUT ASSUMING [sht, shy, shx]

    [sht, shz, shc, shy, shx] = kspace.shape

    # this is now x, y, z, c, t
    bart_kspace = np.transpose(kspace, (4, 3, 1, 2, 0))

    try:
        spatial_denoiser = SpatialDenoiser.load_from_checkpoint(checkpoint_path=SPATIAL_DENOISER_PATH)
        temporal_denoiser = TemporalDenoiser.load_from_checkpoint(checkpoint_path=TEMPORAL_DENOISER_PATH)
    except:
        spatial_denoiser = SpatialDenoiser()
        temporal_denoiser = TemporalDenoiser()

    solver = LowRankSolver(spatial_denoiser=spatial_denoiser, temporal_denoiser=temporal_denoiser, lambda_reg=1e-1, cascades=5)

    solver.to(device)
    solver.eval()
    bart_kspace = torch.from_numpy(bart_kspace).to(device)

    # kspace no z, t, c, h, w
    bart_kspace = np.transpose(bart_kspace, (2, 4, 3, 0, 1))
    
    # estimate sense maps
    maps = []
    for i in range(bart_kspace.shape[0]):
        maps.append(espirit(bart_kspace[[i], 0, ...].permute(0, 2, 3, 1).to(device), 8, 16, 0.001, 0.99, device))
    maps = torch.cat(maps, 0)
    
    # solve for k-space
    with torch.no_grad():
        recon_k_space = solver(bart_kspace, bart_kspace != 0, maps)

    # z, t, y, x
    recon_images = root_sum_of_squares(ifft_2d_img(recon_k_space), 2)

    # t, z, y, x
    recon_images = recon_images.permute(1, 0, 2, 3)

    return recon_images.cpu().numpy()

import h5py
if __name__ == '__main__':
    file = '/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/Aorta/ValidationSet/UnderSample_Task2/P001/aorta_sag_kus_ktGaussian24.mat'
    data = None
    with h5py.File(file, 'r') as fr: 
        data = fr['kus'][:]
    
    data = data['real'] + 1j * data['imag']

    recon = lowrank(data, mask=None)

