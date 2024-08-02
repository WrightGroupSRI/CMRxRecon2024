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
from .bartutils import bart 

def zf_recon(kspace=None, mask=None, **kwargs):

    # transpose to bart format

    bart_kspace = np.transpose(kspace, (4, 3, 1, 2, 0))

    zf_recon = bart(1, "fft -i -u 3", bart_kspace)

    return bart(1, "rss 8", zf_recon)[:, :, :, 0, :]

