#!/usr/bin/env python3

###############################################################
# CMRXRECON MODULE
# functions for diffferent reconstruction modes
#
# Calder Sheagren
# University of Toronto
# calder.sheagren@sri.utoronto.ca
# Date: April 27, 2024
###############################################################

from .zf_recon import zf_recon
from .bart_recon import *
from functools import partial

itsense_recon = partial(bart_recon, command="pics -g -d5 -l2 -r 0.1")
espirit_recon = partial(bart_recon, command="pics -g -d5 -R W:7:0:0.01")
