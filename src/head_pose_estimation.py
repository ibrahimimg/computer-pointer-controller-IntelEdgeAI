from openvino.inference_engine import IENetwork,IEPlugin
import os
import cv2
import argparse
import sys

class HeadPoseEstimationModel:
    '''
    Class for HP Model.
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
        outputs = self.exec_net.requests[0].outputs
        pose_angles = self.preprocess_output(outputs)
        
        return pose_angles

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
        # Parse head pose detection results
        angle_p_fc = outputs["angle_p_fc"][0][0]
        angle_y_fc = outputs["angle_y_fc"][0][0]
        angle_r_fc = outputs["angle_r_fc"][0][0]
        return [angle_y_fc,angle_p_fc,angle_r_fc]