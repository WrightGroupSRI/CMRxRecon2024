#!/usr/bin/env python3

###############################################################
# SCRIPT TITLE
# script functionality
#
# Calder Sheagren
# University of Toronto
# calder.sheagren@sri.utoronto.ca
# Date: August 10, 2024
###############################################################

import numpy as np
import matplotlib.pyplot as plt

import os
import sys

import glob
import time

import cfl
from utils import writemat, try_dir

def main(data_path, dest_path):
    try_dir(dest_path)

    multicoil = os.path.join(data_path, "MultiCoil")
    dest_multicoil = try_dir(os.path.join(dest_path, "MultiCoil"))

    for modality in os.listdir(multicoil):
        patient_dir = os.path.join(multicoil, modality, "ValidationSet", "Task2")
        dest_modality = try_dir(os.path.join(dest_multicoil, modality))
        dest_valset = try_dir(os.path.join(dest_modality, "ValidationSet"))
        dest_patientdir = try_dir(os.path.join(dest_valset, "Task2"))
        for pt in os.listdir(patient_dir):
            dest_pt = try_dir(os.path.join(dest_patientdir, pt))
            for f in glob.glob(os.path.join(patient_dir, pt, "*.cfl")):
                x = np.abs(cfl.readcfl(f.split(".cfl")[0]))
                dest_f = os.path.join(dest_path, "MultiCoil", modality, "ValidationSet", "Task2", pt, f.split("/")[-1].split(".cfl")[0])
                writemat(key="img4ranking", data=x, path=dest_f)

    return True

if __name__ == '__main__': 
    main(sys.argv[1], sys.argv[2])



