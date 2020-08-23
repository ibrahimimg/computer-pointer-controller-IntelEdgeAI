'''
This is a a base class which can be used by all models
it contains some methods that can be used by sub classes
'''

from openvino.inference_engine import IENetwork, IEPlugin
from openvino.inference_engine import IECore
import os
import cv2
import argparse
import sys
import numpy as np


class Helper:
    '''
    Class Helper
    '''
    def __init__(self, model, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_structure = model
        self.model_weights = os.path.splitext(model)[0]+".bin"
        self.device = device
        self.extensions = extensions
        # cpu extensions are added automatically in later versions 2020+ 
        # you can specify the custom path
        self.to_add_extension = False
        self.model_name = ""
        self.inference_time = 0.0

        # Initialize the inference ingine
        self.ie = IEPlugin(device=self.device)
        #self.ie = IECore()
        #d_version = IECore().get_versions(self.device)
        
        if self.extensions and "CPU" in self.device:
            try:
                self.ie.add_cpu_extension(self.extensions)
                self.to_add_extension = True
            except:
                raise AttributeError("can not add CPU extension, does x paths exists?")

        try:
            self.net = IENetwork(model=self.model_structure, weights=self.model_weights)
        
        except Exception:
            raise ValueError("Could not Initialise the network. does model path exists")

        # Get the input layer
        self.input_name=next(iter(self.net.inputs))
        self.input_shape=self.net.inputs[self.input_name].shape
        
        self.output_name=next(iter(self.net.outputs))
        # self.output_shape=self.net.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.check_model()
        # Loading network to device
        self.exec_net = self.ie.load(network=self.net, num_requests=1)
        #self.exec_net = IECore().load_network(self.net, self.device)

    def check_model(self):
        #  Check for supported layers in model ###
        supported_layers = self.ie.get_supported_layers(self.net)
        not_supported_layers = list(filter(lambda l: l not in supported_layers, map(lambda l: l, self.net.layers.keys())))
        if len(not_supported_layers) != 0:
            print("Following layers are not supported for device: ", self.device)
            print(', '.join(not_supported_layers))
            exit(1)
        return True
        
    def req_get(self, r_type, input_dict, outputs=None):
        # number of requests is 1, index 0
        if r_type == "async":
            # Start an asynchronous request, number of requests is 1, id = 0
            self.exec_net.start_async(request_id=0,inputs=input_dict)
            # Wait for the request to be complete.
            status = self.exec_net.requests[0].wait(-1)
        elif r_type == "sync":    
            self.exec_net.requests[0].infer(inputs=input_dict)
            status = 0

        if status==0:
            # Get the result and return outputs
            if outputs:
                return self.exec_net.requests[0].outputs[outputs]
            return self.exec_net.requests[0].outputs
            
    def preprocess_input(self, image, **kw):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        # Pre-process the image as needed #
        
        if len(kw)==1:
            _width_l,_height_l=self.net.inputs['left_eye_image'].shape[2:]
            _width_r,_height_r=self.net.inputs['right_eye_image'].shape[2:]
            
            left_eye_image = image
            right_eye_image = kw['right_eye_image']
            try:
                le = self.get_pr_im(left_eye_image, _width_l, _height_l)
                re = self.get_pr_im(right_eye_image, _width_r, _height_r)
            except cv2.error:
                return None
            return le, re

        else:
            _width=self.input_shape[3]
            _height=self.input_shape[2]
            return self.get_pr_im(image,_width,_height)

    def get_pr_im(self, image, _width, _height):
        p_image = cv2.resize(image, (_width, _height))
        p_image = p_image.transpose((2,0,1))
        p_image = p_image.reshape(self.input_shape[0], self.input_shape[1], _height, _width)
        return p_image
        
    
