import cv2
import numpy as np
import logging
import time
from math import cos, sin, pi
from helper import Helper

class GazeEstimationModel(Helper):
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model, device='CPU', extensions=None):
        super().__init__(model,device,extensions)
        self.input_shape_l=self.net.inputs['left_eye_image'].shape
        self.input_shape_r=self.net.inputs['right_eye_image'].shape
        self.output_names = [n for n in self.net.outputs.keys()]
        self.output_name = self.output_names

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        Helper.load_model(self)

    def predict(self, left_eye_image, right_eye_image, hpa):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        # process the input
        LR_input_result = self.preprocess_input(left_eye_image, right_eye_image = right_eye_image)
        if LR_input_result is None:
            return None, None
        
        le_img_processed, re_img_processed = LR_input_result
        self.p_frame = LR_input_result
        
        _input_dict = {
            'head_pose_angles':hpa,
            'left_eye_image':le_img_processed,
            'right_eye_image':re_img_processed
        }
        
        # Perform inference on the frame and get output results
        result = self.req_get(r_type="sync", input_dict=_input_dict)

        outputs = self.preprocess_output(result, hpa)
        new_mouse_coord = outputs[0]
        gaze_vector = outputs [1]
        return new_mouse_coord, gaze_vector

    def preprocess_output(self, outputs, hpe_angles):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        gaze_vector = list(outputs[self.output_names[0]])[0]
  
        cos_value = cos(hpe_angles[2]*pi/180.0)
        sin_value = sin(hpe_angles[2]*pi/180.0)
        
        new_x = gaze_vector[0]*cos_value+gaze_vector[1]*sin_value
        new_y = (-gaze_vector[0])*sin_value+gaze_vector[1]*cos_value
        
        return (new_x,new_y), gaze_vector
