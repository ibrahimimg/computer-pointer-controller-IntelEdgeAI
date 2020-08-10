from helper import Helper
import cv2

class HeadPoseEstimationModel(Helper):
    '''
    Class for HP Model.
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
        outputs = self.exec_net.requests[0].outputs
        pose_angles = self.preprocess_output(outputs)
        
        return pose_angles

    def preprocess_input(self, image):
        return Helper.preprocess_input(self, image)

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