#!/usr/bin/bash

sudo docker build -t debug -f Dockerfile .
sudo docker run --gpus all -v /hdd/Data/CMRxRecon/:/input -v /hdd/Data/CMRxReconOutput/:/output --rm debug
