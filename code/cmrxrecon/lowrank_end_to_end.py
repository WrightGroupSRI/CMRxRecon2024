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

import numpy as np
from cmrxrecon.dl.lowrank_varnet import LowRankLightning
import torch
from cmrxrecon.utils import ifft_2d_img, root_sum_of_squares
from torchvision.transforms import Compose
from cmrxrecon.dl.AllContrastDataModule import NormalizeKSpace, ZeroPadKSpace
from .smaps import calc_maps
import copy

def lowrank_e2e(kspace: np.ndarray, device, lambda_reg=1e-1, weights_dir=None):
    ######## NOT SURE THE MASK DIMENSIONS BUT ASSUMING [sht, shy, shx]

    [sht, shz, shc, shy, shx] = kspace.shape


    solver = LowRankLightning.load_from_checkpoint(weights_dir)
    norm = NormalizeKSpace()
    pad = ZeroPadKSpace()

    solver.to(device)
    solver.eval()

    bart_kspace = np.transpose(kspace, (4, 3, 1, 2, 0))
    sens_maps = calc_maps(kspace=bart_kspace)
    first_map = np.transpose(sens_maps[..., 0], (2, 3, 1, 0))

    kspace = torch.from_numpy(kspace)

    # kspace now z, t, c, h, w
    kspace = torch.permute(kspace, (1, 0, 2, 3, 4))

    # this is now x, y, z, c, t
    mask = kspace != 0
    
    # estimate sense maps
    # maps = calc_espirit(bart_kspace, device)
    maps = torch.from_numpy(first_map).unsqueeze(1)
    #torch.save(maps, 'maps.pt')
    # maps = torch.load('maps.pt')

    k_space = [] 
    mask = []
    padded_maps = []

    for i in range(kspace.shape[0]): 
        values = norm((kspace[i], kspace[i] != 0, maps[i]))
        values = pad(values)
        k_space.append(values[0])
        mask.append(values[1])
        padded_maps.append(values[2])

    k_space = torch.stack(k_space, dim=0).to(device)
    mask  = torch.stack(mask, dim=0).to(device)
    padded_maps = torch.stack(padded_maps, dim=0).to(device)
    
    # solve for k-space
    recon_k_space = copy.deepcopy(k_space)
    with torch.no_grad():
        for i in range(recon_k_space.shape[0]):
            recon_k_space[i, ...] = solver(k_space[[i], ...].cfloat(), k_space[[i], ...] != 0, padded_maps[[i], ...].cfloat())

    # z, t, y, x
    recon_images = root_sum_of_squares(ifft_2d_img(recon_k_space), coil_dim=2)

    return np.transpose(recon_images.cpu().numpy(), (3, 2, 0, 1))

"""
import h5py
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from torchvision.utils import make_grid
if __name__ == '__main__':
    file = '/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/Aorta/ValidationSet/UnderSample_Task2/P001/aorta_sag_kus_ktGaussian8.mat'
    data = None
    with h5py.File(file, 'r') as fr: 
        data = fr['kus'][:]
    
    data = data['real'] + 1j * data['imag']

    recon = lowrank(data, mask=None, device='cuda')
    plt.imshow(make_grid(torch.from_numpy(recon[:, [0], :, :]), nrow=3)[0], cmap='gray', vmax=recon[:, [0], :, :].max()/4)
    plt.savefig('lowrank_recon')
"""

