import cv2
import numpy as np
from helper import Helper
import time
class FacialLandmarksDetectionModel(Helper):
    '''
    Class for the Facial Landmarks Detection Model.
    '''
    def __init__(self, model, device='CPU', extensions=None):
        super().__init__(model,device,extensions)
        self.current_frame = None
        self.model_name = "Facial Landmarks Detection"

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
        result = self.req_get(r_type="sync", input_dict=_input_dict, outputs=self.output_name)
        self.inference_time=time.time()-inference_start

        left_eye, right_eye = self.preprocess_output(result)
        return left_eye, right_eye

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        l_eye_x, l_eye_y, r_eye_x, r_eye_y = map(lambda i : outputs[0][i][0][0], [0,1,2,3])
        coords = [l_eye_x,l_eye_y,r_eye_x,r_eye_y]

        # post processing
        h,w = self.current_frame.shape[0:2]
        coords = coords * np.array([w, h, w, h])
        coords = coords.astype(np.int32)
 
        # resize the box
        le_xmin, le_ymin, re_xmin, re_ymin = map(lambda i : coords[i]-15, [0,1,2,3])
        le_xmax, le_ymax, re_xmax, re_ymax = map(lambda i : coords[i]+15, [0,1,2,3])
        
        # cropping
        left_eye =  self.current_frame[le_ymin:le_ymax, le_xmin:le_xmax]
        right_eye = self.current_frame[re_ymin:re_ymax, re_xmin:re_xmax]
        
        return left_eye, right_eye
