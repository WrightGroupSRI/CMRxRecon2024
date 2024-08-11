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
from cmrxrecon.utils import pad_to_shape, crop_to_shape
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

    mask = kspace != 0
    
    # estimate sense maps
    # maps = calc_espirit(bart_kspace, device)
    maps = torch.from_numpy(first_map).unsqueeze(1)
    #torch.save(maps, 'maps.pt')
    # maps = torch.load('maps.pt')

    k_space = [] 
    mask = []
    padded_maps = []

    scaling_factor = k_space.abs().amax((1, 2, 3, 4), keepdim=True)

    k_space, original_pad = pad_to_shape(kspace/scaling_factor, [256, 512])
    padded_maps, original_pad = pad_to_shape(maps, [256, 512])

    # solve for k-space
    recon_k_space = copy.deepcopy(k_space)
    with torch.no_grad():
        for i in range(recon_k_space.shape[0]):
            recon_k_space[i, ...] = solver(k_space[[i], ...].cfloat(), k_space[[i], ...] != 0, padded_maps[[i], ...].cfloat())

    recon_k_space = crop_to_shape(recon_k_space, original_pad) 
    recon_k_space = recon_k_space * scaling_factor
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
    file = '/home/kadotab/scratch/MICCAIChallenge2024/ChallengeData/MultiCoil/Aorta/ValidationSet/UnderSample_Task2/P002/aorta_sag_kus_ktRadial12.h5'

    data = None
    with h5py.File(file, 'r') as fr: 
        data = fr['kus'][:]
    
    data = data['real'] + 1j * data['imag']

    recon = lowrank(data, mask=None, device='cuda')
    plt.imshow(make_grid(torch.from_numpy(recon[:, [0], :, :]), nrow=3)[0], cmap='gray', vmax=recon[:, [0], :, :].max()/4)
    plt.savefig('lowrank_recon')
"""

