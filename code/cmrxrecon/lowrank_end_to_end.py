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
from cmrxrecon.dl.lowrank_varnet import LowRankLightning
import torch
from cmrxrecon.espirit import espirit
from cmrxrecon.utils import ifft_2d_img, root_sum_of_squares
from torchvision.transforms import Compose
from cmrxrecon.dl.AllContrastDataModule import NormalizeKSpace, ZeroPadKSpace

SPATIAL_DENOISER_PATH = '/home/kadotab/scratch/cmrxrecon_checkpoints/2024-08-07_13_lowrank_epoch=9-val/loss=0.00-val/ssim=0.91-v1.ckpt'


def calc_espirit(bart_kspace, device):
    maps = []
    for i in range(bart_kspace.shape[0]):
        map = espirit(bart_kspace[[i], 0, ...].permute(0, 2, 3, 1).to(device), 5, 16, 0.0001, 0.99, device)
        maps.append(map.permute(0, 3, 1, 2))
    maps = torch.cat(maps, 0)
    return maps

def lowrank(kspace: np.ndarray, mask=None, device='cpu', lambda_reg=1e-1):
    ######## NOT SURE THE MASK DIMENSIONS BUT ASSUMING [sht, shy, shx]

    [sht, shz, shc, shy, shx] = kspace.shape


    solver = LowRankLightning.load_from_checkpoint(SPATIAL_DENOISER_PATH)
    norm = NormalizeKSpace()
    pad = ZeroPadKSpace()

    solver.to(device)
    solver.eval()
    kspace = torch.from_numpy(kspace)

    # kspace now z, t, c, h, w
    kspace = torch.permute(kspace, (1, 0, 2, 3, 4))

    # this is now x, y, z, c, t
    mask = kspace != 0
    
    # estimate sense maps
    #maps = calc_espirit(bart_kspace, device)
    #maps = maps.unsqueeze(1)
    #torch.save(maps, 'maps.pt')
    maps = torch.load('maps.pt')

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
    with torch.no_grad():
        recon_k_space = solver(k_space, k_space != 0, padded_maps)

    # z, t, y, x
    recon_images = root_sum_of_squares(ifft_2d_img(recon_k_space), coil_dim=2)

    # t, z, y, x
    recon_images = recon_images.permute(1, 0, 2, 3)
    print(recon_images.shape)
    

    return recon_images.cpu().numpy()

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


