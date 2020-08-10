from helper import Helper
import cv2
import numpy as np

class FacialLandmarksDetectionModel(Helper):
    '''
    Class for FL Model.
    '''
    def __init__(self, model, device, extensions=None):
        super().__init__(model,device,extensions)

    def load_model(self):
        Helper.load_model(self)

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''

        self.current_frame = image
        # process the current frame
        self.pr_frame = self.preprocess_input(self.current_frame)
        input_dict={self.input_name: self.pr_frame}
        # get output results
        self.exec_net.requests[0].infer(inputs=input_dict)
        outputs = self.exec_net.requests[0].outputs[self.output_name]

        coords = self.preprocess_output(outputs)
        h,w = self.current_frame.shape[0:2]
        coords = coords * np.array([w, h, w, h])
        coords = coords.astype(np.int32)

        le_xmin=coords[0]-10
        le_ymin=coords[1]-10
        le_xmax=coords[0]+10
        le_ymax=coords[1]+10
        
        re_xmin=coords[2]-10
        re_ymin=coords[3]-10
        re_xmax=coords[2]+10
        re_ymax=coords[3]+10
        
        left_eye =  self.current_frame[le_ymin:le_ymax, le_xmin:le_xmax]
        right_eye = self.current_frame[re_ymin:re_ymax, re_xmin:re_xmax]
        eye_coords = [[le_xmin,le_ymin,le_xmax,le_ymax], [re_xmin,re_ymin,re_xmax,re_ymax]]
        return left_eye, right_eye, eye_coords


    def preprocess_input(self, image):
        return Helper.preprocess_input(self, image)

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        outputs=outputs[0]
        leye_x = outputs[0][0][0]
        leye_y = outputs[1][0][0]
        reye_x = outputs[2][0][0]
        reye_y = outputs[3][0][0]
        
        return leye_x, leye_y, reye_x, reye_y

