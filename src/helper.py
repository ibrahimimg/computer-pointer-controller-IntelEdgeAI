'''
This is a a base class which can be used by all models
it contains some methods that may be used by sub classes
'''

from openvino.inference_engine import IENetwork, IEPlugin
from openvino.inference_engine import IECore, get_version
import os
import cv2
import argparse
import sys
import numpy as np


class Helper:
    '''
    Class Helper
    '''
    def __init__(self, model, device, extensions):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_structure = model
        self.model_weights = os.path.splitext(model)[0]+".bin"
        self.device = device
        
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

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        # Loading network to device
        self.exec_net = self.ie.load(network=self.net, num_requests=1)

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