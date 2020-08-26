import cv2
import numpy as np
import logging
import time
from helper import Helper

class FaceDetectionModel(Helper):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model, device='CPU', extensions=None, threshold=0.60):
        super().__init__(model,device,extensions)
        self.threshold = threshold
        self.current_frame = None
        self.model_name = "Face Detection"

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
        self.height,self.width,_=self.current_frame.shape
        # process the current frame
        self.pr_frame = self.preprocess_input(self.current_frame)
        _input_dict={self.input_name: self.pr_frame}

        inference_start = time.time()
        # Perform inference on the frame and get output results
        result = self.req_get(r_type="sync",input_dict=_input_dict,outputs=self.output_name)
        self.inference_time=time.time()-inference_start

        outputs = self.preprocess_output(result)
        if outputs is None:
            return None
        cropped_face, coords = outputs
        return cropped_face, coords

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        coords=list()
        #for box in outputs[0][0]:
        for b in range (len(outputs[0][0])):
            box = outputs[0][0][b]
            confidence = box[2]
            if confidence > self.threshold:
                coords_list = [box[3],box[4],box[5], box[6]]
                coords.append(coords_list)

                if self.show:
                    # preprocessing for showing detection
                    xmin,xmax = map(lambda b : int(b*self.width), [box[3],box[5]])
                    ymin,ymax = map(lambda b : int(b*self.height), [box[4],box[6]])
                    cv2.rectangle(self.current_frame,(xmin-15,ymin-15), (xmax+15,ymax+15), (0, 255, 0) , 2)
            
        # postprocessing
        if len(coords) == 0:
            # No Face detected
            return None
        else:
            if len(coords) > 1:
                logging.getLogger().warning("more than one faces detected!")
                logging.getLogger().info("....one face will be used")
            # Take the first face detected
            coords = coords [0]
            
        h,w = self.current_frame.shape[0], self.current_frame.shape[1]
        coords = coords * np.array([w, h, w, h])
        coords = coords.astype(np.int32)
        cropped_face = self.current_frame[coords[1]:coords[3], coords[0]:coords[2]]
        return cropped_face, coords