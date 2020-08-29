# Computer Pointer Controller

| Details               |                 |
|-----------------------|-----------------|
| Programming Language  |  Python 3.7     |
| Operating System      |  Ubuntu 18.04   |

![pointer-controller](./images/demo.gif)

Computer Pointer Controller (CPC) is A.I application to control the computer mouse pointer movement using eye's gaze, the project use Inference Engine API from Intel's OpenVINO ToolKit with python implementation. The four different pretrained models will run in the same machine; the gaze estimation model will use the results from the previous models to find the new mouse coordinates, the app then change the mouse pointer position accordingly.

## How it Works

The app will use a media file or webcam as input and capture frames, each frame is sent to face detection model, after some preprocessing it returns the cropped face, the cropped face is then sent to the head post estimation and facial landmarks detection models for further processing, the cropped eyes (left and right eyes) are returned by facial landmarks detection model while head pose estimation model returns head pose angles, these results are passed to the gaze estimation model where the returned coordinates will be used as new values for pointer and hence move the mouse position according to result.

### Pipeline

- The flow of data will look like this
    ![pipeline-diagram](./images/pipeline.png)


## Requirements

### Hardware

* 6th to 10th generation Intel® Core™ processor with Iris® Pro graphics or Intel® HD Graphics.
* OR use of Intel® Neural Compute Stick 2 (NCS2) or FPGA device.

### Software

* Intel® Distribution of OpenVINO™ toolkit
* Python3.7 with PIP
* you can use anaconda

## Project Directory Structure
* 
    ![directory-structure](./images/dir-structure.png)

| Name |  Description    |
|------|-----------------|
|.     | the root folder which contains folders, log files, scripts and source code files|
|bin   | this folder contain a media files, a video and image that can  be used as input for this app|
|images| This folder contain images like pipeline and screeshots used in README.md file|
|intel | this directory contains pretrained models of different precision|
|src   | This is directory where the source code of this application are saved|
|*.sh  | bash scripts used to automate task, they include written bash commands|
|*.log | stdout and log info of the app. also contain the performance result of models and hardware|
|requirements.txt|list of python libraries required for this project, to be installed using pip|
|README.md| this file. project details, how to set up and run the code etc.

## Project Set Up and Installation

### 1. Install Intel® Distribution of OpenVINO™ toolkit
Before you run this application you should have OpenVINO™ toolkit installed in your machine, for more information about OpenVINO visit this [page](https://docs.openvinotoolkit.org/latest/index.html)

*_OpenVINO 2020.1 is recommended for this project_*

*_you can use anaconda (tested on version 1.9.7)_*

### 2. Launch your Terminal
* before you run this application lauch your terminal, usually by pressing CRTL+ALT+T
* cd to project directory
* paste the following commands

### 3. Source OpenVINO™ environment
You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit 

    source /opt/intel/openvino/bin/setupvars.sh -pyver 3.7

### 4. Download Pretrained models: 

    chmod +x download_models.sh
    ./download_models.sh

this file contain a written commands to download all the models required for this project from intel open model zoo.

these pretrained models are saved in the current working directory. you can view the file here [download_mods.sh](download_models.sh)
 
### 5. Create Virtual Environment

* Create a virtual environment cpc with python version 3.7

    `virtualenv cpc -p python3.7`

    if you don't have virtual environment installed, you should install it first:

    `pip install virtualenv`

* Activate the cpc environment

    `source cpc/bin/activate`

* install required libraries in virtualenv

    `pip install -r requirements.txt`

![demo](./images/venv.png)


## Documentation
this section may help you better understand the command line arguments that this project requires and how to run a basic demo.

__make sure you are in project directory__

### Source OpenVINO™ environment
before you run this app You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit 

    source /opt/intel/openvino/bin/setupvars.sh -pyver 3.7

### usage 

    python3.7 main.py [-h] -fd FACE_DETECTION_MODEL -fl FACIAL_LANDMARK_MODEL -hp
            HEAD_POSE_MODEL -ge GAZE_ESTIMATION_MODEL -i INPUT [-d DEVICE]
            [-pt PROB_THRESHOLD] [-x EXTENSION] [-sh SHOW_OUTPUT]


### required arguments:

    -fd FACE_DETECTION_MODEL, --face_detection_model FACE_DETECTION_MODEL
                            specify the Path to Face Detection model xml file
    -fl FACIAL_LANDMARK_MODEL, --facial_landmark_model FACIAL_LANDMARK_MODEL
                            specify the Path to Facial Landmarks Detection model
                            xml file
    -hp HEAD_POSE_MODEL, --head_pose_model HEAD_POSE_MODEL
                            specify the Path to Head Pose Estimation model xml
                            file
    -ge GAZE_ESTIMATION_MODEL, --gaze_estimation_model GAZE_ESTIMATION_MODEL
                            specify the Path to Gaze Estimation model xml file
    -i INPUT, --input INPUT
                            specify input, use media file or type cam to use your
                            webcam

### optional arguments:

    -d DEVICE, --device DEVICE
                            specify the target device to run inference on: CPU,
                            GPU, FPGA or MYRIAD (for NCS2)
    -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                            specify probability threshold for model to detect the
                            face accurately from the frame
    -x EXTENSION, --extension EXTENSION
                            specify path to CPU extension file, if applicable, for
                            OpenVINO version < 2020
    -sh SHOW_OUTPUT, --show_output SHOW_OUTPUT
                            specify whether to Show Visualization of output
                            results for a model e.g: 'fd fl ge' or use 'all'
               

### Running Demo
This is how to run a basic demo this project.

automatically:
if you are in project root folder, type this command in terminal:

    bash demo.sh

or paste the following commands:

*make sure you are in **src** folder*. on the project root folder, type 

`cd src`

`python3.7 main.py -fd ../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fl ../intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml -hp ../intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -ge ../intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -d CPU -i ../bin/demo.mp4 -s "all"`

![demo](./images/cpc.png)

## Benchmarks

The performance of all possible combination of model precisions is calculated and the log file is saved. see [stdout.log](./stdout.log).

Hardware device used: Intel® Core™ i5-6200U CPU. \
you can check it out on other hardware devices as well. use this command:

`bash run_combination.sh device_name`

example :

`bash run_combination.sh GPU`


## Results

In stdout.log file, we can see that the models loading time, frames per second and total inference time are all depends on the model precisions used. using other hardware devices such as GPU or VPU will vary different results.

fd=face detection model, fl=facial landmarks, hp=head pose e and ge=gaze estimation

let us take three combinations for example here is the results:

1. MODELS COMBINATION: fd, fl, hp, ge :  FP32-INT1 FP32 FP32 FP32

    models loading time  :  0.44 \
    frames per seconds   :  9.5 \
    total inference time :  6.21

2. MODELS COMBINATION: fd, fl, hp, ge :  FP32-INT1 FP16 FP16 FP16

    models loading time  :  0.47 \
    frames per seconds   :  12.79 \
    total inference time :  4.61 

3. MODELS COMBINATION: fd, fl, hp, ge :  FP32-INT1 FP32-INT8 FP32-INT8 FP32-INT8

    models loading time  :  1.83 \
    frames per seconds   :  9.24 \
    total inference time :  6.39 


- Here we run our application using different precision of models, usually inference time in low precision models is faster. the model with high precision say FP32 will require more compution than FP16 and INT8 models, loading time is one time event and low value means it tooks less time to load a models, the frames per second (FPS) and inference time of the low precision models are better, however, there are some trade-offs with accuracy when using lower precision, FP32 have a higher accuracy than others e.g: FP16. FP32 for example can be quantized into INT8 this will allow us to get faster inference but  with some drops in accuracy.

- the best precision combination is "FP32-INT1 FP16 FP16 FP16". FP32-INT1 will be used for face detection model while FP16 will be used for other models. by doing this we will maintain our system resources while having a good accuracy that is enough.

## Stand Out Suggestions

- User can shut off the output video by pressing q key. This will close the cv2 opened window and show a log info like this:

    ```
    INFO:root:changing mouse position... moving
    INFO:root:video window closed... to exit app, press CTRL+Z
    INFO:root:changing mouse position... moving
    INFO:root:changing mouse position... moving
    ```

- This app can handle image, video and webcam (i have added a code to preprocess the webcam since it is backwards/inverted).


### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

*I used synchronous inference in my project and implemented asynchronous inference in my code as well.*

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.

- The app used 60% as default probability threshold to ensure best result during face detection.

- if the face is not detected either due to low light or other situations, the app output log info in terminal telling the user: No face detected. The app then skip the frame without feeding data to other models.

- user can also specify the probability threshold based on their need.

- when the app detects more than one face in frame, it will show a log warning in terminal and tell the user only one face will be used.

- If errors occur in other models after face detection results, the app shows info to the user, the frame is skipped and the mouse position remains unchanged.


## Conclusion

In this project, we used the gaze detection model to control the mouse pointer of computer. The Gaze Estimation model was used to estimate the gaze of the user's eyes and change the mouse pointer position accordingly. This project also demonstrate the ability to run multiple models in the same machine and coordinate the flow of data between those models.

    Ibrahim Ishaka - 2020