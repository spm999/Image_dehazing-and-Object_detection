import cv2
import numpy as np

'''
   HazeImg: Hazy input image
   Transmission: estimated transmission
   A: estimated airlight
   delta: fineTuning parameter for dehazing --> default = 0.85
   return: result --> Dehazed image
  '''
def Estemate_airlight(HazeImg, AirlightMethod, windowSize):
    if(AirlightMethod.lower() == 'sss'):
        A = []
        if(len(HazeImg.shape) == 3):
            for ch in range(len(HazeImg.shape)):
                kernel = np.ones((windowSize, windowSize), np.uint8)  # creates the array
                minImg = cv2.erode(HazeImg[:, :, ch], kernel)#########################
                A.append(int(minImg.max()))
        else:
            kernel = np.ones((windowSize, windowSize), np.uint8)
            minImg = cv2.erode(HazeImg, kernel)
            A.append(int(minImg.max()))
    return(A)
