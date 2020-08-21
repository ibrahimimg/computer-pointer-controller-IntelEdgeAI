"""
COMPUTER POINTER CONTROLLER
-IBRAHIM ISHAKA 
-JULY 2020
"""

import os
import sys
import time
import os
import logging
import cv2
import numpy as np
import argparse

from mouse_controller import MouseController
from input_feeder import InputFeeder

from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksDetectionModel
from gaze_estimation import GazeEstimationModel
from head_pose_estimation import HeadPoseEstimationModel

def build_args():
    """
    Parse command line arguments.
    :return: parser
    """
    parser = argparse.ArgumentParser()
    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    help = {
        'fd': "specify the Path to Face Detection model xml file",
        'fl': "specify the Path to Facial Landmarks Detection model xml file",
        'hp': "specify the Path to Head Pose Estimation model xml file",
        'ge': "specify the Path to Gaze Estimation model xml file",
        'i' : "specify input, use media file or type cam to use your webcam",
        'd' : "specify the target device to run inference on: CPU, GPU, FPGA or MYRIAD (for NCS2)",
        'pt': "specify probability threshold for model to detect the face accurately from the frame "
    }
        
    required.add_argument("-fd", "--face_detection_model", required=True, help=help['fd'], type=str)
    required.add_argument("-fl", "--facial_landmark_model", required=True, help=help['fl'], type=str)
    required.add_argument("-hp", "--head_pose_model", required=True, help=help['hp'], type=str)
    required.add_argument("-ge", "--gaze_estimation_model", required=True, help=help['ge'], type=str)
    required.add_argument("-i", "--input", required=True, help=help['i'], type=str)
    
    optional.add_argument("-d", "--device", required=False, default="CPU", help=help['d'], type=str)
    optional.add_argument("-pt", "--prob_threshold", required=False, default=0.6, help=help['pt'], type=float)
    
    return parser

def main():
    # Grab command line args
    args = build_args().parse_args()
    # Config Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    #os.system('clear')
    logger.info("\n==========<COMPUTER POINTER CONTROLLER>==========\n")
    
    # initialize model object for each class
    FDModel = FaceDetectionModel(model=args.face_detection_model, device=args.device, extensions=None, threshold=args.prob_threshold)
    FLDModel = FacialLandmarksDetectionModel(model=args.facial_landmark_model, device=args.device, extensions=None)
    HPEModel = HeadPoseEstimationModel(model=args.head_pose_model, device=args.device,  extensions=None)
    GEModel = GazeEstimationModel(model=args.gaze_estimation_model, device=args.device,  extensions=None)
    
    models = {
        'fd' : FDModel,
        'fl' : FLDModel,
        'hp' : HPEModel,
        'ge' : GEModel
    }

    logger.info("Loading Models")
    for k in models:
        # load model
        models[k].load_model()

    logger.info("Done. All models are loaded")

	# setting for mouse controller
    _precision =  "medium"
    _speed = "fast"
    mouse_controller = MouseController(precision = _precision , speed=_speed)

    # verify and handle input stream
    input_source= args.input
    input_feeder = None
    input_type = ""
    if input_source.lower() !="cam":
        # check if input file exist
        if os.path.exists(input_source) and os.path.isfile(input_source):
            input_type = "video"
            input_feeder = InputFeeder(input_type = input_type, input_file = input_source)
        else:
            logger.error("Input file is not a file, or does't exist")
            sys.exit(1)
    elif input_source.lower() == "cam":
        input_type="cam"
        input_feeder = InputFeeder(input_type = input_type)
    
    input_feeder.load_data()
    frame_count = 0

    for flag, frame in input_feeder.next_batch():
        if flag is False:
            # no frame to read
            break
        frame_count = frame_count + 1
        key_pressed = cv2.waitKey(60)
        
        if input_source=='cam':
            # preprocess frame as webcam is backwards/inverted
            frame = cv2.flip(frame,1)
            
        cv2.imshow(input_type, cv2.resize(frame,(500,500)))
        face_detection_result = FDModel.predict(frame)
        # The prediction result should return None, if no face detected
        if face_detection_result is None:
            if not window_closed:
                cv2.imshow(input_type, cv2.resize(frame,(500,500)))
            logger.info("NO FACE DETECTED... skipping")
            continue
        cropped_face = face_detection_result[0]
        face_coords = face_detection_result[1]
        hp_result = HPEModel.predict(cropped_face)
        left_eye, right_eye = FLDModel.predict(cropped_face)
        new_mouse_coords, gaze_vector = GEModel.predict(left_eye, right_eye, hp_result)
            
        try:
            x,y = new_mouse_coords
        except:
            logger.error("unable to get mouse coordinates for current frame\nReading Next Frame...")
            continue

        if new_mouse_coords is None:
            # Error during LR_eyes processing
            continue

        elif frame_count % 5 == 0:
            logger.info("changing mouse position... moving")
            mouse_controller.move(x,y)

        # Break if escape key pressed
        if key_pressed==27:
            break
       
    # Release the capture and destroy any OpenCV window
    input_feeder.close()
    cv2.destroyAllWindows()

    logger.info("Stream ended !")
    
    if frame_count == 0:
        logger.error("Unable to handle Unsupported file ")
        sys.exit(1)

if __name__ == '__main__':
    main()
    
