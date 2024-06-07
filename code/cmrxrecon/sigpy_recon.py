#!/usr/bin/env python3

###############################################################
# SIGPY RECON
# so I don't have to compile bart
#
# Calder Sheagren
# University of Toronto
# calder.sheagren@sri.utoronto.ca
# Date: June 07, 2024
###############################################################

import numpy as np

import sigpy as sp
import sigpy.mri as mr

def zf_recon(kspace=None, mask=None, device=None):

    # print(kspace.shape)

    if len(kspace.shape) < 5:
        # blackblood = True
        coil_ax = 1
    else:
        # blackblood = False
        coil_ax = 2

    return np.sum(np.abs(sp.ifft(kspace, axes=(-1, -2)))**2, axis=coil_ax)**0.5


def sense_recon(kspace=None, mask=None, device=None):
    pass

def espirit_recon(kspace=None, mask=None):
    pass
