#!/usr/bin/bash

docker build -t debug -f Dockerfile .
docker run --gpus all -v /hdd/Usamp_Scan_Data/CMRxRecon_Data/:/input -v /hdd/Usamp_Scan_Data/CMRxReconSubmission/FlipX:/output --rm debug
