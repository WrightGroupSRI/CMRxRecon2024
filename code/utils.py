#!/usr/bin/env python3

###############################################################
# UTILS.PY
# Utility functions
#
# Calder Sheagren
# University of Toronto
# calder.sheagren@sri.utoronto.ca
# Date: April 27, 2024
###############################################################

import hdf5storage as hf
import h5py
import numpy as np

import os

def try_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def loadmat(key=None, path=None):
    assert(path), "Please pass in filepath"
    with h5py.File(path) as f:
        keys = list(f.keys())
        if len(keys) == 1:
            return np.array(f[keys[0]])

def writemat(key=None, data=None, path=None):
    assert(key), "Please pass in key"
    assert(data.ndim > 0), "Please pass in data"
    assert(path), "Please pass in path"
    hf.savemat(path, {key:data}, appendmat=False)
    return path

def post_crop(x):
    st, sz, sy, sx = x.shape
    # fix
    return x[:, :, sy // 4:3 * sy // 4, sx // 3: 2 * sx // 3]
