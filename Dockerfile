# Copy to Dockerfile

# Specify the base image
FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Toronto

RUN apt-get update

# Install GPU drivers (NVIDIA instructions)
RUN apt-get install -y wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
RUN mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/12.2.1/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.1-535.86.10-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.1-535.86.10-1_amd64.deb
RUN cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update
RUN apt-get -y install cuda
RUN ln -s /usr/local/cuda/lib64 /usr/local/cuda/lib

# Install BART with Cuda (Based on Kelvin Chow Python-ISMRMRD Server)
RUN apt-get install -y git cmake g++ libfftw3-dev liblapacke-dev libpng-dev gfortran
RUN mkdir -p /opt/code

# BART (static linked)
RUN cd /opt/code                                   && \
    git clone https://github.com/mrirecon/bart.git --branch v0.7.00 && \
    cd bart                                        && \
    make CUDA=1 CUDA_BASE=/usr/local/cuda/ CUDA_LIB=lib64 -j $(nproc)                       && \
    make install

ENV PYTHONPATH=/opt/code/bart/python

# install python
RUN apt-get install -y python3 python3-pip

# mount volumes
VOLUME /input
VOLUME /output

# install dependencies from requirements.txt
COPY code/requirements.txt /
RUN pip install -r /requirements.txt

# Copy the code directory to /app
COPY code /app

# DO NOT EDIT THE FLLOWING LINES
COPY *_run.py /
COPY submitter.json /
# You can run more commands when the container start by 
# editing docker-entrypoint.sh
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/bin/bash", "/docker-entrypoint.sh"]
