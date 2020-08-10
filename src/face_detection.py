from helper import Helper
from helper import Helper
import cv2
import numpy as np

class FaceDetectionModel(Helper):
    '''
    Class for FD Model.
    '''
    def __init__(self, model, device, extensions, threshold):
        super().__init__(model,device,extensions)
        self.threshold=threshold

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
        self.exec_net.requests[0].infer(input_dict)
        outputs = self.exec_net.requests[0].outputs[self.output_name]
        coords = self.preprocess_output(outputs)
        if len(coords)==0:
            return None,None
        else:
            # take one face
            coords=coords[0]
        h,w = self.current_frame.shape[0:2]
        coords = coords * np.array([w, h, w, h])
        coords = coords.astype(np.int32)
        cropped_face = self.current_frame[coords[1]:coords[3], coords[0]:coords[2]]
        return cropped_face, coords

    def preprocess_input(self, image):
        return Helper.preprocess_input(self, image)

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        coordinates=list()
        #for box in outputs[0][0]:
        for b in range (len(outputs[0][0])):
            box = outputs[0][0][b]
            confidence = box[2]
            if confidence > self.threshold:
                p1=(box[3],box[4])
                p2=(box[5],box[6])
                coordinates.append([p1[0],p1[1],p2[0],p2[1]])
                
        return coordinates
