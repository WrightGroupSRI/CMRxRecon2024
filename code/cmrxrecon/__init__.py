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
from .unet_recon import *
from .lowrank_end_to_end import lowrank_e2e
from functools import partial

itsense_recon = partial(bart_recon, command="pics -g -d5 -l2 -r 0.1")
espirit_recon = partial(bart_recon, command="pics -g -d5 -R W:7:0:0.01")
temporal_espirit_recon = partial(bart_recon, command="pics -d5 -R T:7:0:0.005 -R L:32:0:0.001") # not very good
