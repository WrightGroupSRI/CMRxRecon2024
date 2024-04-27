#!/usr/bin/env python3

###############################################################
# MAIN.PY
# CMRxRecon 2024 Challenge Code 
#
# Calder Sheagren
# University of Toronto
# calder.sheagren@sri.utoronto.ca
# Date: April 26, 2024
###############################################################

import numpy as np
import torch

import os
import sys

import glob
import time
from datetime import datetime

from utils import *
import cmrxrecon as cxr

import argparse

# temporary
import cfl

def main(args):

    # path_to_model_weights = args.weights_dir

    match args.recon_mode:
        case "zf":
            recon_func = cxr.zf_recon
        case "pi":
            recon_func = cxr.itsense_recon
        case "cs":
            recon_func = cxr.espirit_recon

    match args.challenge:
        case "validation":
            dataset = "ValidationSet"

    data_dir = os.path.join(args.input_dir, "MultiCoil", "Mapping", dataset)

    _ = try_dir(args.predict_dir)
    _ = try_dir(os.path.join(args.predict_dir, "MultiCoil")) 
    _ = try_dir(os.path.join(args.predict_dir, "MultiCoil", "Mapping"))
    output_dir = try_dir(os.path.join(args.predict_dir, "MultiCoil", "Mapping", dataset))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    for a in ["04", "08", "10"]:
        acc_factor = f"AccFactor{a}"

        acc_dir = os.path.join(data_dir, acc_factor)

        acc_dir_output = try_dir(os.path.join(output_dir, acc_factor))

        if not os.path.exists(acc_dir):
            print(acc_dir, "does not exist")
            continue

        for pt in glob.glob(os.path.join(acc_dir, "P*")):

            pt_dir_output = try_dir(os.path.join(acc_dir_output, pt.split("/")[-1]))

            T1map = os.path.join(pt, "T1map.mat")

            T2map = os.path.join(pt, "T2map.mat")

            T1map_mask = os.path.join(pt, "T1map_mask.mat")

            T2map_mask = os.path.join(pt, "T2map_mask.mat")

            for (kspace_path, mask_path) in zip([T1map, T2map], [T1map_mask, T2map_mask]):

                mask = loadmat(key=f"mask{a}", path=mask_path)

                kspace = loadmat(key=f"kspace_sub{a}", path=kspace_path)

                img = recon_func(kspace=kspace, mask=mask)

                # temporary
                cfl.writecfl(os.path.join(pt_dir_output, kspace_path.split("/")[-1].split(".mat")[0]), img)

                writemat(key="img4ranking", data=img, path=os.path.join(pt_dir_output, kspace_path.split("/")[-1]))


if __name__ == '__main__': 

    parser = argparse.ArgumentParser(
            prog="CMRxRecon2024",
            description="3D+t Reconstruction",
            epilog="Have a good day!"
            )

    parser.add_argument("--input_dir")
    parser.add_argument("--predict_dir")
    parser.add_argument("--weights_dir")

    parser.add_argument("--recon_mode") # zf, pi, cs for now
    parser.add_argument("--challenge") # train, validation, test

    args = parser.parse_args()

    main(args)

