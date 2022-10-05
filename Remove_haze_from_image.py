import cv2
import numpy as np
import copy

def Remove_haze_from_image(HazeImg, Transmission, A, delta):##########################333
    '''
     HazeImg: Hazy input image
     Transmission: estimated transmission
     A: estimated airlight
     delta: fineTuning parameter for dehazing --> default = 0.85
     return: result --> Dehazed image
    '''
    epsilon = 0.0001
    Transmission = pow(np.maximum(abs(Transmission), epsilon), delta)
    HazeCorrectedImage = copy.copy(HazeImg)
    if(len(HazeImg.shape) == 3):
        for ch in range(len(HazeImg.shape)):
            temp = ((HazeImg[:, :, ch].astype(float) - A[ch]) / Transmission) + A[ch]
            temp = np.maximum(np.minimum(temp, 255), 0)
            HazeCorrectedImage[:, :, ch] = temp
    else:
        temp = ((HazeImg.astype(float) - A[0]) / Transmission) + A[0]
        temp = np.maximum(np.minimum(temp, 255), 0)
        HazeCorrectedImage = temp
    return(HazeCorrectedImage)