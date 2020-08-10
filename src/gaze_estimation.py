from helper import Helper
import cv2
import numpy as np
import math

class GazeEstimationModel(Helper):
    '''
    Class for GE Model.
    '''
    def __init__(self, model, device, extensions=None):
        super().__init__(model,device,extensions)
        self.input_shape_l=self.net.inputs['left_eye_image'].shape
        self.input_shape_r=self.net.inputs['right_eye_image'].shape


    def load_model(self):
        Helper.load_model(self)

    def predict(self, left_eye_image, right_eye_image, hpa):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''

        self.current_frame_l = left_eye_image
        self.current_frame_r = right_eye_image
        
        # process the current frame
        self.pr_frame = self.preprocess_input(left_eye_image, right_eye_image)
        
        le_img_processed, re_img_processed = self.preprocess_input(left_eye_image.copy(), right_eye_image.copy())
        self.exec_net.requests[0].infer({
            'head_pose_angles':hpa, 
            'left_eye_image':le_img_processed, 
            'right_eye_image':re_img_processed})
        outputs = self.exec_net.requests[0].outputs[self.output_name]
        new_mouse_coord, gaze_vector = self.preprocess_output(outputs, hpa)

        return new_mouse_coord, gaze_vector

    def preprocess_input(self, le_image, re_image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        # Pre-process the image as needed #
        _width=self.input_shape_l
        _height=self.input_shape_r
        
        le_res_image = cv2.resize(le_image, (self.input_shape_l[3], self.input_shape_l[2]))
        re_res_image = cv2.resize(re_image, (self.input_shape_r[3], self.input_shape_r[2]))

        le_p_image = np.transpose(np.expand_dims(le_res_image,axis=0), (0,3,1,2))
        re_p_image = np.transpose(np.expand_dims(le_res_image,axis=0), (0,3,1,2))
        
        return le_p_image, re_p_image
        #raise NotImplementedError

    def preprocess_output(self, outputs, hpa):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''

        gaze_vector = outputs[0]
        #gaze_vector = gaze_vector / cv2.norm(gaze_vector)
        rollValue = hpa[2] #angle_r_fc output from HeadPoseEstimation model
        cosValue = math.cos(rollValue * math.pi / 180.0)
        sinValue = math.sin(rollValue * math.pi / 180.0)
        
        newx = gaze_vector[0] * cosValue + gaze_vector[1] * sinValue
        newy = -gaze_vector[0] *  sinValue+ gaze_vector[1] * cosValue
        return (newx,newy), gaze_vector
