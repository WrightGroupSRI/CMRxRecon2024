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
# import matplotlib.pyplot as plt

import os
import sys

import glob
import time

from bart import bart

import argparse

def main(path):
    return True

if __name__ == '__main__': 

    parser = argparse.ArgumentParser(
            prog="CMRxRecon2024",
            description="Multicontrast Reconstruction",
            epilogue="Have a good day!"
            )

    parser.add_argument("--input_dir")
    parser.add_argument("--predict_dir")
    parser.add_argument("--weights_dir")

    print(parser.input_dir)
    print(parser.predict_dir)
    print(parser.weights_dir)




