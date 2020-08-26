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
import pyautogui

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
    
    required.add_argument("-fd", "--face_detection_model", required=True, type=str, 
                          help="specify the Path to Face Detection model xml file")
    required.add_argument("-fl", "--facial_landmark_model", required=True, type=str, 
                          help="specify the Path to Facial Landmarks Detection model xml file")
    required.add_argument("-hp", "--head_pose_model", required=True, type=str, 
                          help="specify the Path to Head Pose Estimation model xml file")
    required.add_argument("-ge", "--gaze_estimation_model", required=True, type=str, 
                          help="specify the Path to Gaze Estimation model xml file")
    required.add_argument("-i", "--input", required=True, type=str, 
                          help="specify input, use media file or type cam to use your webcam")
 
    optional.add_argument("-d", "--device", required=False, default="CPU", type=str, 
                          help="specify the target device to run inference on: CPU, GPU, FPGA or MYRIAD (for NCS2)")
    optional.add_argument("-pt", "--prob_threshold", required=False, default=0.6, type=float,
                          help="specify probability threshold for model to detect the face accurately from the frame ")
    optional.add_argument("-x", "--extension", required=False, default=None, type=str, 
                          help="specify path to CPU extension file, if applicable, for OpenVINO version < 2020")
    optional.add_argument("-sh", "--show_output", required=False, default="", type=str, 
                          help="specify whether to Show Visualization of output results for a model e.g: 'fd fl ge' or use 'all'")
    return parser

def main():
    # Grab command line args
    args = build_args().parse_args()
    # Config Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    #os.system('clear')
    print("\n")
    logger.info("starting app ...")
    print("\n==========<COMPUTER POINTER CONTROLLER>==========")
    print("============>(c) Ibrahim Ishaka 2020<============\n")
    
    # initialize model object for each class
    FDModel = FaceDetectionModel(model=args.face_detection_model, device=args.device, extensions=args.extension, threshold=args.prob_threshold)
    FLDModel = FacialLandmarksDetectionModel(model=args.facial_landmark_model, device=args.device, extensions=args.extension)
    HPEModel = HeadPoseEstimationModel(model=args.head_pose_model, device=args.device,  extensions=args.extension)
    GEModel = GazeEstimationModel(model=args.gaze_estimation_model, device=args.device,  extensions=args.extension)
    
    models = {
        'fd' : FDModel,
        'fl' : FLDModel,
        'hp' : HPEModel,
        'ge' : GEModel
    }

    models_loading_time = 0
    for k in models:
        # load model
        logger.info("Loading {} Model".format(models[k].model_name))
        model_loading_start = time.time()
        models[k].load_model()
        model_loading_finish = (time.time()-model_loading_start)
        models_loading_time = models_loading_time + model_loading_finish
        logger.info("time taken to load Model: {:.3f}secs".format(model_loading_finish))

        # check if model output visualization is specified in sh arg
        if k in args.show_output or args.show_output=='all':
            models[k].show = True
        logger.info("show {} outputs: {} \n".format(models[k].model_name,models[k].show))

    logger.info("time taken to load All Models: {:.3f}secs\n".format(models_loading_time))

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
            image_formats=[".png",".jpg",".bmp",".jpeg"]
            is_image = [True for x in image_formats if input_source.endswith(x)]
            if is_image:
                input_type = "image"
            else:
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
    total_inference_time_all = 0
    window_closed=False
    
    for flag, frame in input_feeder.next_batch():
        if flag is False:
            # no frame to read
            break
        frame_count = frame_count + 1
        key_pressed = cv2.waitKey(60)
        
        if input_source=='cam':
            # preprocess frame as webcam is backwards/inverted
            frame = cv2.flip(frame,1)
            
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
        
        total_inference_time = 0
        for key in models:
            total_inference_time = total_inference_time + models[key].inference_time
            total_inference_time_all = total_inference_time_all + total_inference_time
        
        #uncomment the following line to see the inference time for each frame
        #logger.info("Inference Time : {:.3f}".format(total_inference_time))
        
        try:
            x,y = new_mouse_coords
        except:
            logger.error("unable to get mouse coordinates for current frame\nReading Next Frame...")
            continue
        
        if GEModel.show == True:
            GEModel.show_gaze(left_eye, right_eye, gaze_vector)
        if HPEModel.show == True:
            frame = HPEModel.show_hp(frame, hp_result)

        if new_mouse_coords is None:
            # Error during LR_eyes processing
            continue

        '''
        wait on before moving mouse again
        this is recomended to avoid failsafe exception
        but you change this setting
        '''
        if input_type=="image":
            cv2.imshow(input_type, cv2.resize(frame,(500,500)))
            mouse_controller.move(x,y)
            break
            
        if frame_count % 5 == 0:
            try:
                logger.info("changing mouse position... moving")
                mouse_controller.move(x,y)
            except pyautogui.FailSafeException:
                logger.error("safe exception From pyautogui")
                continue
                
        if not window_closed:
            cv2.imshow(input_type, cv2.resize(frame,(500,500)))        
        
        # Break if escape key pressed
        if key_pressed==27:
            break
            
        # close the OpenCV window if q key pressed    
        if key_pressed == ord('q'):
            window_closed=True
            cv2.destroyWindow(input_type)
            logger.info(input_type+" window closed... to exit app, press CTRL+Z")
            
    if frame_count != 0:
        # Release the capture and destroy any OpenCV window
        input_feeder.close()
        cv2.destroyAllWindows()
    
        logger.info("Stream ended !")

        fps=round(frame_count/total_inference_time_all,2)
        print ("\n==========SUMMARY===========")
        print("models loading time  : ",round(models_loading_time,2))
        print("frames per seconds   : ",fps)
        print("total inference time : ",round(total_inference_time_all,2))
        print ("============================")

    else:
        logger.error("Unable to handle Unsupported file ")
        sys.exit(1)

if __name__ == '__main__':
    main()
    
