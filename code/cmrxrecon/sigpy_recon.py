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

def zf_recon(kspace=None, mask=None, sp_device=None):

    # print(kspace.shape)

    if len(kspace.shape) < 5:
        # blackblood = True
        coil_ax = 1
    else:
        # blackblood = False
        coil_ax = 2

    zf_rss = np.sum(np.abs(sp.ifft(kspace, axes=(-1, -2)))**2, axis=coil_ax)**0.5
    return np.transpose(zf_rss, (-1, -2, -3, 0))

def calc_smaps(kspace=None, mask=None, device=None):

    sht, shz, shc, shy, shx = kspace.shape

    maps = np.zeros((shz, shc, shy, shx), dtype=np.complex_)

    for z in range(shz):

        ksp = kspace[0, z, ...]

        maps[z, ...] = mr.app.EspiritCalib(ksp, device=device, calib_width=16).run().get()

    return maps

def sense_recon(kspace=None, mask=None, sp_device=0, device=None):
    mps_ = calc_smaps(kspace, mask, sp_device)

    sht, shz, shc, shy, shx = kspace.shape

    img = np.zeros((sht, shz, shy, shx), dtype=np.complex_)

    for z in range(shz):

        for t in range(sht):

            img[t, z, ...] = mr.app.SenseRecon(kspace[t, z, ...], mps_[z, ...], lamda=0.01, device=sp_device).run().get()

    return np.transpose(img, axes=(-1, -2, -3, 0))


def espirit_recon(kspace=None, mask=None, sp_device=0, device=None):
    mps_ = calc_smaps(kspace, mask, sp_device)

    sht, shz, shc, shy, shx = kspace.shape

    img = np.zeros((sht, shz, shy, shx), dtype=np.complex_)

    for z in range(shz):

        for t in range(sht):

            img[t, z, ...] = mr.app.L1WaveletRecon(kspace[t, z, ...], mps_[z, ...], 1e-6, device=sp_device).run().get()

    return np.transpose(img, axes=(-1, -2, -3, 0))
