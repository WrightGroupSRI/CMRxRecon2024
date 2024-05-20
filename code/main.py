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
import copy
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
        case "training":
            dataset = "TrainingSet"
            modalities = ["Aorta", "Cine", "Mapping", "Tagging"]
            sample = "FullSample"
        case "validation":
            dataset = "ValidationSet"
            modalities = ["Aorta", "BlackBlood", "Cine", "Flow2d", "Mapping", "Tagging"]



    _ = try_dir(args.predict_dir)
    _ = try_dir(os.path.join(args.predict_dir, "MultiCoil")) 

    for mod in modalities:
        data_dir = os.path.join(args.input_dir, "MultiCoil", mod, dataset)

        _ = try_dir(os.path.join(args.predict_dir, "MultiCoil", mod))
        _ = try_dir(os.path.join(args.predict_dir, "MultiCoil", mod))
        _ = try_dir(os.path.join(args.predict_dir, "MultiCoil", mod, dataset))
        output_dir = try_dir(os.path.join(args.predict_dir, "MultiCoil", mod, dataset, sample))

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        for pt in glob.glob(os.path.join(data_dir, sample, "P*")):

            pt_folder = pt.split("/")[-1]

            pt_dir_output = try_dir(os.path.join(output_dir, pt_folder))

            for mat_file in glob.glob(os.path.join(pt, "*.mat")):

                print("Processing", mat_file)

                _ksp = loadmat(path=mat_file)

                kspace = _ksp['real'] + 1j * _ksp['imag']

                print("KSPACE:", kspace.shape)

                prefix = mat_file.split("/")[-1].split(".mat")[0]

                mask1_list = os.listdir(os.path.join(data_dir, "Mask_Task1", pt_folder))

                masks1 = [os.path.join(data_dir, "Mask_Task1", pt_folder, m) for m in mask1_list if prefix in m]

                mask2_list = os.listdir(os.path.join(data_dir, "Mask_Task2", pt_folder))

                masks2 = [os.path.join(data_dir, "Mask_Task2", pt_folder, m) for m in mask2_list if prefix in m]

                masks = masks1 + masks2

                for m in masks:
                    print("Processing", m)
                    mask = np.array(loadmat(path=m))
                    print("MASK:", mask.shape)

                    try:
                        masked_kspace = kspace * np.broadcast_to(mask, kspace.shape)
                    except:
                        masked_kspace = copy.deepcopy(kspace)
                        for i in range(kspace.shape[1]):
                            for j in range(kspace.shape[2]):
                                masked_kspace[:, i, j, :, :] *= mask

                    print("KSP MASK", masked_kspace.shape)

                    img = recon_func(kspace=masked_kspace, mask=mask)

                    if "Uniform" in m:

                        R = m.split(".mat")[0].split("Uniform")[-1]

                        fname = f"{prefix}_kus_Uniform{R}"

                    else:

                        SamplingR = m.split(".mat")[0].split("kt")[-1]

                        fname = f"{prefix}_kus_kt{SamplingR}"

                    dest_path = os.path.join(pt_dir_output, fname)

                    cfl.writecfl(dest_path, post_crop(img))
                    # fix naming conventions
                    writemat(key="img4ranking", data=post_crop(img), path=dest_path)

                    break # masks

                break # mat file

            break # patients

        break # modalities


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

