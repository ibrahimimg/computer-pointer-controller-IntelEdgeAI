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
from argparse import ArgumentParser

from mouse_controller import MouseController
from input_feeder import InputFeeder

from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksDetectionModel
from gaze_estimation import GazeEstimationModel
from head_pose_estimation import HeadPoseEstimationModel


#time_for_file_name=datetime.datetime.now().replace(microsecond=0).isoformat()

def get_args():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fd", "--face_detection_model", required=True, type=str,
                        help="Path to Face Detection model xml.")
    parser.add_argument("-fl", "--facial_landmark_model", required=True, type=str,
                        help="Path to Facial Landmark Detection model xml.")
    parser.add_argument("-hp", "--head_pose_model", required=True, type=str,
                        help="Path to Head Pose Estimation model xml.")
    parser.add_argument("-ge", "--gaze_estimation_model", required=True, type=str,
                        help="Path to Gaze Estimation model xml.")

    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or enter cam for webcam")
    
    parser.add_argument("-pt", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for model to detect the face accurately from the video frame.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. ")
    
    args = parser.parse_args()
    return args

def main():
    logger = logging.getLogger()
    # Grab command line args
    args = get_args()
    input_source = args.input
    inputFeeder = None
    
    if input_source.lower() !="cam":
        if os.path.exists(input_source) and os.path.isfile(input_source):
            inputFeeder = InputFeeder("video",input_source)
        else:
            print ("Input file is not a file, or does't exist")
            sys.exit(1)
    else:
        inputFeeder = InputFeeder("cam")
    

    fdm = FaceDetectionModel(args.face_detection_model, args.device,None, args.prob_threshold)
    fldm = FacialLandmarksDetectionModel(args.facial_landmark_model, args.device)
    gem = GazeEstimationModel(args.gaze_estimation_model, args.device)
    hpem = HeadPoseEstimationModel(args.head_pose_model, args.device)
    
    mc = MouseController('medium','fast')
    
    inputFeeder.load_data()
    fdm.load_model()
    fldm.load_model()
    hpem.load_model()
    gem.load_model()

    frame_count = 0
    for ret, frame in inputFeeder.next_batch():
        if not ret:
            break
        frame_count+=1
        if frame_count%5==0:
            cv2.imshow('video',cv2.resize(frame,(500,500)))

        key = cv2.waitKey(60)
        croppedFace, face_coords = fdm.predict(frame)
        
        if croppedFace is None:
            logger.error("Unable to detect the face.")
            if key==27:
                break
            continue
            
        hp_out = hpem.predict(croppedFace)
        left_eye, right_eye, eye_coords = fldm.predict(croppedFace)
        new_mouse_coord, gaze_vector = gem.predict(left_eye, right_eye, hp_out)

        if frame_count%5==0:
            mc.move(new_mouse_coord[0],new_mouse_coord[1])    
        if key==27:
                break

    logger.error("VideoStream ended...")
    cv2.destroyAllWindows()
    inputFeeder.close()
    

if __name__ == '__main__':
    main()

    
    