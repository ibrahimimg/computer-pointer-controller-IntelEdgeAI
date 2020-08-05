from openvino.inference_engine import IENetwork,IEPlugin
import os
import cv2
import argparse
import sys
import numpy as np

class FacialLandmarksDetectionModel:
    '''
    Class for FL Model.
    '''
    def __init__(self, model, device, extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_structure = model
        self.model_weights = os.path.splitext(model)[0]+".bin"
        self.device = device
        self.extensions = extensions

        # Initialize the inference ingine
        self.ie = IEPlugin(device=self.device)
        try:
            self.net = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        # Get the input layer
        self.input_name=next(iter(self.net.inputs))
        self.input_shape=self.net.inputs[self.input_name].shape
        
        self.output_name=next(iter(self.net.outputs))
        self.output_shape=self.net.outputs[self.output_name].shape
        #raise NotImplementedError

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        # Loading network to device
        self.exec_net = self.ie.load(network=self.net, num_requests=1)
                       
        #raise NotImplementedError

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

        #raise NotImplementedError

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        # Pre-process the image as needed #
        _width=self.input_shape[3]
        _height=self.input_shape[2]
        p_image = cv2.resize(image, (_width, _height))
        p_image = p_image.transpose((2,0,1))
        p_image = p_image.reshape(self.input_shape[0], self.input_shape[1], _height, _width)
        
        return p_image
        #raise NotImplementedError

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

        
        #raise NotImplementedError
