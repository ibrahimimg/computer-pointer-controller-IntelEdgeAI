# Computer Pointer Controller

Computer Pointer Controller (CPC) is A.I application to control the computer mouse pointer movement using eye's gaze, the project use Inference Engine API from Intel's OpenVINO ToolKit with python implementation. The four different pretrained models will run in the same machine; the gaze estimation model will use the results from the previous models to find the new mouse coordinates, the app then change the mouse pointer position accordingly.

## How it Works

The app will use a media file or webcam as input and capture frames, each frame is sent to face detection model, after some preprocessing it returns the cropped face, the cropped face is then sent to the head post estimation and facial landmarks detection models for further processing, the cropped eyes (left and right eyes) are returned by facial landmarks detection model while head pose estimation model returns head pose angles, these results are passed to the gaze estimation model where the returned coordinates will be used as new values for pointer and hence move the mouse position according to result.

## Requirements

### Hardware

* 6th to 10th generation Intel® Core™ processor with Iris® Pro graphics or Intel® HD Graphics.
* OR use of Intel® Neural Compute Stick 2 (NCS2) or FPGA device

### Software

* Intel® Distribution of OpenVINO™ toolkit
* Python3.7 with PIP
* you can use anaconda

## Project Set Up and Installation

### 1. Install Intel® Distribution of OpenVINO™ toolkit
Before you run this application you should have OpenVINO™ toolkit installed in your machine, for more information about OpenVINO visit this [page](https://docs.openvinotoolkit.org/latest/index.html)

*_OpenVINO 2020.1 is recommended for this project_*

*_you can use anaconda for running python (v3.7)_*

### 2. Launch your Terminal
* before you run this application lauch your terminal, usually by pressing CRTL+ALT+T
* cd to project directory
* paste the following commands

### 3. Source OpenVINO™ environment
You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit 

    source /opt/intel/openvino/bin/setupvars.sh -pyver 3.7

### 4. Download Pretrained models:
paste these commands

    cwd=$(pwd)
    cd /opt/intel/openvino/deployment_tools/tools/model_downloader/
    ./downloader.py --name face-detection-adas-binary-0001 -o $cwd
    ./downloader.py --name landmarks-regression-retail-0009 -o $cwd
    ./downloader.py --name head-pose-estimation-adas-0001 -o $cwd
    ./downloader.py --name gaze-estimation-adas-0002 -o $cwd
    cd $cwd

### 5. Create Virtual Environment

* Create a virtual environment cpc with python version 3.7

    `virtualenv cpc -p python3.7`

    if you don't have virtual environment installed, you should install it first:

    `pip install virtualenv`

* Activate the cpc environment

    `source cpc/bin/activate`

* install required libraries in virtualenv

    `pip install -r requirements.txt`


## Documentation
use

`python main.py --help`

to see the available options with description. 

### Running Demo
    
`cd src`

`python3.7 main.py -fd ../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fl ../intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml -hp ../intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -ge ../intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -d CPU -i ../bin/demo.mp4 `


## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
