# Copy to Dockerfile

# Specify the base image
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Toronto

RUN apt update

RUN mkdir -p /opt/code

# install python
RUN apt-get install -y python3 python3-pip

# install bart
RUN apt-get install -y git cmake g++ libfftw3-dev liblapacke-dev libpng-dev gfortran
RUN apt-get install -y bart-cuda

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
