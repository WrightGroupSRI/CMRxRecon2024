#!/usr/bin/env python3

###############################################################
# SCRIPT TITLE
# script functionality
#
# Calder Sheagren
# University of Toronto
# calder.sheagren@sri.utoronto.ca
# Date: July 31, 2024
###############################################################

from .unet import *
import numpy as np
import torch
from .cflutils import writecfl
from .bartutils import bart
from .utils import pad_to_shape, crop_to_shape

def unet_recon(kspace, device, **kwargs):
    st, sz, sc, sy, sx = kspace.shape
    for z in range(sz):
        kspace[:, z, ...] /= np.max(np.abs(kspace[:, z, ...]))
    recon_fft = bart(1, "fft -u -i 3", np.transpose(kspace, (4, 3, 2, 1, 0)))
    recon_rss = bart(1, "rss 8", recon_fft)
    recon_zf = np.transpose(recon_rss[..., 0, :], (3, 2, 1, 0))

    recon = np.zeros((sx, sy, sz, st), dtype=np.complex_)

    ul = UnetLightning.load_from_checkpoint(kwargs.get("weights_dir"), input_channels=1, lr=1e-4)
    model_input = np.expand_dims(np.expand_dims(np.abs(recon_zf), axis=0), axis=0)
    for t in range(st): 
        for z in range(sz):
            pad_input, original_recipe = pad_to_shape(torch.cuda.FloatTensor(model_input[:, :, t, z, :, :]), [256, 512])
            model_output = ul.model(pad_input)
            print(model_output.shape)
            out = np.transpose(model_output, axes=(3, 2, 1, 0)) 
            recon[..., z, t] = np.squeeze(out)

    return recon

if __name__ == '__main__': 
    path = ""
    main(path)



