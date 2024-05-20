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
from bart import bart 

def zf_recon(kspace=None, mask=None):

    # transpose to bart format

    bart_kspace = np.transpose(kspace, (4, 3, 1, 2, 0))

    zf_recon = bart(1, "fft -i -u 3", bart_kspace)

    zf_recon_rss = bart(1, "rss 8", zf_recon)

    print("RSS:", zf_recon_rss.shape)

    # transpose to CMRxRecon format

    zf_recon_final = np.transpose(zf_recon_rss, (4, 2, 3, 1, 0))[:, :, 0, :, :]

    print("FINAL:", zf_recon_final.shape)

    return zf_recon_final
