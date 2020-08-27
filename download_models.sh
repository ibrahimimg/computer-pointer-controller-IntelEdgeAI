#!/bin/bash

# get current working dir
cwd=$(pwd)

# change directory to model_downloader path
cd /opt/intel/openvino/deployment_tools/tools/model_downloader/

# Download models, -o means output directory, = current dir

# 1 Face Detection Model
./downloader.py --name face-detection-adas-binary-0001 -o $cwd

# 2 Facial Landmarks Detection Model
./downloader.py --name landmarks-regression-retail-0009 -o $cwd

# 3 Head Pose Estimation Model
./downloader.py --name head-pose-estimation-adas-0001 -o $cwd

# 4 Gaze Estimation Model
./downloader.py --name gaze-estimation-adas-0002 -o $cwd

# back to current working dir
cd $cwd
