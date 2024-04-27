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
        if key:
            data = f[key]
        else:
            print(f.keys())
        try:
            data = data['real'] + 1j * data['imag']
        except:
            pass

    # data = mat73.loadmat(path)
    # array = data.get(key)
    return data

def writemat(key=None, data=None, path=None):
    assert(key), "Please pass in key"
    assert(data.ndim > 0), "Please pass in data"
    assert(path), "Please pass in path"
    hf.savemat(path, {key:data}, appendmat=False)
    return path
