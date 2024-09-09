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

import os
import sys

import glob
import time

import math

from cflutils import readcfl
from utils import writemat, loadmat, try_dir

def xycrop(image, crop_size):
    sx, sy, sz, st = image.shape
    cx, cy, cz, ct = crop_size
    start_x = math.floor(sx / 2)
    start_y = math.floor(sy / 2)

    return image[start_x + math.ceil(-cx/2):start_x + math.ceil(cx/2), 
                 start_y + math.ceil(-cy/2):start_y + math.ceil(cy/2), :, :]

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

                #ref_fpath = os.path.join(ref_path, "MultiCoil", modality, "ValidationSet", "Task2", pt, f.split("/")[-1].split(".cfl")[0])
                #x_ref = loadmat(key="img4ranking", path=ref_fpath)

                x = np.abs(readcfl(f.split(".cfl")[0]))

                #x_crop = xycrop(x, x_ref.shape)

                dest_f = os.path.join(dest_path, "MultiCoil", modality, "ValidationSet", "Task2", pt, f.split("/")[-1].split(".cfl")[0])

                writemat(key="img4ranking", data=np.transpose(x, (3, 2, 1, 0)), path=dest_f)


    return True

if __name__ == '__main__': 
    #ref_path = sys.argv[3]
    main(sys.argv[1], sys.argv[2])#, ref_path)



