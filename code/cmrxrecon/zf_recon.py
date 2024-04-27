#!/usr/bin/env python3

###############################################################
# SCRIPT TITLE
# script functionality
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

    # transpose to CMRxRecon format

    zf_recon_final = np.squeeze(np.transpose(zf_recon_rss, (4, 2, 1, 0, 3)))

    return zf_recon_final
