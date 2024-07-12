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

def calc_smaps(kspace=None, mask=None, device=None):

    sht, shz, shc, shy, shx = kspace.shape

    maps = np.zeros((shz, shc, shy, shx), dtype=np.complex_)

    for z in range(shz):

        ksp = kspace[0, z, ...]

        maps[z, ...] = mr.app.EspiritCalib(ksp, device=device, calib_width=16).run().get()

    return maps


