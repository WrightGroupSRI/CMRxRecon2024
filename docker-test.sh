#!/usr/bin/bash

sudo docker build -t debug -f Dockerfile .
sudo docker run --gpus all -v /hdd/Data/CMRxRecon/2024/ChallengeData/:/input -v /hdd/Data/CMRxRecon/2024/Output/:/output --rm debug
