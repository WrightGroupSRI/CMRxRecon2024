#!/usr/bin/env python3

###############################################################
# SENSITIVITY MAPS CALCULATION
# for parallel imaging and CS reconstruction
#
# Calder Sheagren
# University of Toronto
# calder.sheagren@sri.utoronto.ca
# Date: April 27, 2024
###############################################################

import numpy as np
from bart import bart

# Function to estimate maps 
def calc_maps(kspace=None, mask=None):

    [shx, shy, shz, shc, sht] = np.shape(kspace)

    array_for_maps_calc = kspace[:, :, :, :, 0]

    maps = np.zeros([shx, shy, shz, shc], dtype=complex)

    # loop over slices
    for i in range(shz):

        maps_kspace = np.expand_dims(array_for_maps_calc[:, :, i, :], axis=0)

        # TODO get the calibration region using the mask (largest connected component?)
        # m_ = bart(1, 'ecalib -S -g -d1 -m2 -a -r1:48:9', maps_kspace)
        m_ = bart(1, 'ecalib -S -d1 -m1 -c 0.99 -t 0.00005 -r 16:16', maps_kspace)

        maps[:, :, i, :] = np.squeeze(m_)

    return maps

