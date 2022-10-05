import cv2
import numpy as np
def Boundary_Constraint(HazeImg, A, C0, C1, windowSze):
    # imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    '''
       HazeImg: Hazy input image
       Transmission: estimated transmission
       A: estimated airlight
       delta: fineTuning parameter for dehazing --> default = 0.85
       return: result --> Dehazed image
      '''
    if(len(HazeImg.shape) == 3):

        t_b = np.maximum((A[0] - HazeImg[:, :, 0].astype(np.float)) / (A[0] - C0),(HazeImg[:, :, 0].astype(np.float) - A[0]) / (C1 - A[0]))
        t_g = np.maximum((A[1] - HazeImg[:, :, 1].astype(np.float)) / (A[1] - C0),(HazeImg[:, :, 1].astype(np.float) - A[1]) / (C1 - A[1]))
        t_r = np.maximum((A[2] - HazeImg[:, :, 2].astype(np.float)) / (A[2] - C0),(HazeImg[:, :, 2].astype(np.float) - A[2]) / (C1 - A[2]))

        MaxVal = np.maximum(t_b, t_g, t_r)
        transmission = np.minimum(MaxVal, 1)
    else:
        transmission = np.maximum((A[0] - HazeImg.astype(np.float)) / (A[0] - C0),(HazeImg.astype(np.float) - A[0]) / (C1 - A[0]))
        transmission = np.minimum(transmission, 1)

    kernel = np.ones((windowSze, windowSze), np.float)  #array creation
    transmission = cv2.morphologyEx(transmission, cv2.MORPH_CLOSE, kernel=kernel)
    return(transmission)




