#!/usr/bin/env python3

###############################################################
# ITERATIVE SENSE RECONSTRUCTION
# parallel imaging recon 
#
# Calder Sheagren
# University of Toronto
# calder.sheagren@sri.utoronto.ca
# Date: April 27, 2024
###############################################################

import numpy as np
from .bartutils import bart 

from .smaps import maps

def bart_recon(kspace=None, mask=None, command=None, sp_device=None):

    [sht, shz, shc, shy, shx] = kspace.shape

    bart_kspace = np.transpose(kspace, (4, 3, 1, 2, 0))

    sens_maps = maps(bart_kspace)

    bart_recon_image = np.zeros((shx, shy, shz, sht), dtype=complex)

    # loop over slices
    for i in range(shz):

        bart_slice_kspace = np.expand_dims(bart_kspace[:, :, i, :, :], axis=0)

        bart_recon_kspace = np.expand_dims(bart_slice_kspace, axis=-2)

        bart_maps = np.expand_dims(sens_maps[:, :, i, :], axis=0)

        aux_recon = bart(1, command, bart_recon_kspace, bart_maps)

        bart_recon_image[:, :, i, :] = aux_recon[0, :, :, 0, 0, :]

    # transpose to CMRxRecon format
    # bart_recon_reshaped = np.transpose(bart_recon_image, (3, 2, 1, 0))

    return bart_recon_image
