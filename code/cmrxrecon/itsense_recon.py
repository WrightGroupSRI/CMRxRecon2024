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
from bart import bart 

from .smaps import maps

def itsense_recon(kspace=None, mask=None):

    [sht, shz, shc, shy, shx] = kspace.shape

    bart_kspace = np.transpose(kspace, (4, 3, 1, 2, 0))

    sens_maps = maps(bart_kspace)

    itsense_recon_image = np.zeros((shx, shy, shz, sht), dtype=complex)
    # loop over slices
    for i in range(shz):

        itsense_slice_kspace = np.expand_dims(bart_kspace[:, :, i, :, :], axis=0)
        itsense_kspace = np.expand_dims(itsense_slice_kspace, axis=-2)

        itsense_maps = np.expand_dims(sens_maps[:, :, i, :], axis=0)

        aux_recon = bart(1, "pics -g -l2 -r0.1 -d5", itsense_kspace, itsense_maps)

        itsense_recon_image[:, :, i, :] = aux_recon[0, :, :, 0, 0, :]

    # transpose to CMRxRecon format

    itsense_recon_reshaped = np.transpose(itsense_recon_image, (3, 2, 1, 0))
    return itsense_recon_reshaped
