#!/bin/bash
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.7
python3.7 ./src/main.py -fd ./intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml \
                        -fl ./intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml \
                        -hp ./intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml \
                        -ge ./intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml \
                        -d CPU \
                        -i ./bin/demo.mp4 \
                        -s "all" 