#!/usr/bin/env python3

###############################################################
# ZERO-FILLED RECONSTRUCTION
# Bart Fourier reconstruction 
#
# Calder Sheagren
# University of Toronto
# calder.sheagren@sri.utoronto.ca
# Date: April 27, 2024
###############################################################

import numpy as np
#from bart import bart 

def zf_recon(kspace=None, mask=None):

    # transpose to bart format

    try:
        bart_kspace = np.transpose(kspace, (4, 3, 1, 2, 0))
        blackblood = False
    except ValueError:
        bart_kspace = np.transpose(kspace, (3, 2, 1, 0))
        blackblood = True

    zf_recon = bart(1, "fft -i -u 3", bart_kspace)

    if not blackblood:
        zf_recon_rss = bart(1, "rss 8", zf_recon)
    else:
        zf_recon_rss = bart(1, "rss 4", zf_recon)

    print("RSS:", zf_recon_rss.shape)

    # transpose to CMRxRecon format

    if not blackblood:
        zf_recon_final = np.transpose(zf_recon_rss, (4, 2, 3, 1, 0))[:, :, 0, :, :]
    else:
        zf_recon_final = np.transpose(zf_recon_rss, (3, 2, 1, 0))[:, 0, :, :]

    print("FINAL:", zf_recon_final.shape)

    return zf_recon_final
