#!/usr/bin/env python3

###############################################################
# ESPIRIT RECONSTRUCTION
# parallel imaging and compressed sensing recon 
#
# Calder Sheagren
# University of Toronto
# calder.sheagren@sri.utoronto.ca
# Date: April 27, 2024
###############################################################

import numpy as np
from bart import bart 

from .smaps import maps

def espirit_recon(kspace=None, mask=None):

    [sht, shz, shc, shy, shx] = kspace.shape

    bart_kspace = np.transpose(kspace, (4, 3, 1, 2, 0))

    sens_maps = maps(bart_kspace)

    espirit_recon_image = np.zeros((shx, shy, shz, sht), dtype=complex)
    # loop over slices
    for i in range(shz):

        espirit_slice_kspace = np.expand_dims(bart_kspace[:, :, i, :, :], axis=0)
        espirit_kspace = np.expand_dims(espirit_slice_kspace, axis=-2)

        espirit_maps = np.expand_dims(sens_maps[:, :, i, :], axis=0)

        aux_recon = bart(1, "pics -g -e -R W:6:0:0.05", espirit_kspace, espirit_maps)

        espirit_recon_image[:, :, i, :] = aux_recon[0, :, :, 0, 0, :]

    # transpose to CMRxRecon format

    espirit_recon_reshaped = np.transpose(espirit_recon_image, (3, 2, 1, 0))
    print(espirit_recon_reshaped.shape)

    return espirit_recon_reshaped
