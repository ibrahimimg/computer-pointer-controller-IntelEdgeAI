import cv2
import numpy as np
from helper import Helper
import time

class HeadPoseEstimationModel(Helper):
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self, model, device='CPU', extensions=None):
        super().__init__(model,device,extensions)
        self.current_frame = None
        self.show = False
        self.model_name = "Head Pose Estimation"

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        Helper.load_model(self)

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        self.current_frame = image
        # process the current frame
        self.pr_frame = self.preprocess_input(self.current_frame)
        _input_dict={self.input_name: self.pr_frame}

        inference_start = time.time()
        # Perform inference on the frame and get output results
        result = self.req_get(r_type="sync", input_dict=_input_dict)
        self.inference_time=time.time()-inference_start

        pose_angles = self.preprocess_output(result)
        return pose_angles

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # Parse head pose detection results
        angle_names = ["angle_y_fc","angle_p_fc","angle_r_fc"]
        yaw,pitch,roll = map(lambda a : outputs[a][0][0], angle_names)

        return [yaw,pitch,roll]