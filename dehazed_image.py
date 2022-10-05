import cv2
import numpy as np

from Estemate_airlight import Estemate_airlight
from Boundary_Constraint import Boundary_Constraint
from CalTransmission import CalTransmission
from Remove_haze_from_image import Remove_haze_from_image
from Resizing_image import Resizing_image

# if __name__ == '__main__':
#     HazeImg = cv2.imread('Images/image14.jpg')
#
#     # To Resize image the image, call the Resizing_image function
#     HazeImg = Resizing_image(HazeImg)
#
#     # Estimate Airlight
#     windowSze = 10
#     AirlightMethod = 'sss'
#     A = Estemate_airlight(HazeImg, AirlightMethod, windowSze)
#
#     # Calculate Boundary Constraints
#     windowSze = 3
#     C0 = 20  # Default value = 20
#     C1 = 300  # Default value = 300
#     Transmission = Boundary_Constraint(HazeImg, A, C0, C1, windowSze)  #########77
#
#     # Refine estimate of transmission
#     regularize_lambda = 0.05  # Default value = 1 --> Regularization parameter, the more this  value, the closer to the original patch wise transmission
#     sigma = 0.8
#     Transmission = CalTransmission(HazeImg, Transmission, regularize_lambda, sigma)  # Using contextual information
#
#     # Perform DeHazing
#     HazeCorrectedImg = Remove_haze_from_image(HazeImg, Transmission, A, 0.99)
#
#     cv2.imshow('Original', HazeImg)
#     cv2.imshow('Result', HazeCorrectedImg)
#     cv2.waitKey(0)
#     print(HazeImg.shape)
#     # print(A)
#     cv2.imwrite('output_Images/The_dehazed_image.jpg', HazeCorrectedImg)
def dehazed_imagee(img):
    # A="Image/"
    HazeImg = cv2.imread(img)
    #To Resize image the image, call the Resizing_image function
    #HazeImg=Resizing_image(HazeImg)

    # Estimate Airlight
    windowSze = 10
    AirlightMethod = 'sss'
    A = Estemate_airlight(HazeImg, AirlightMethod, windowSze)

    # Calculate Boundary Constraints
    windowSze = 3
    C0 = 20  # Default value = 20
    C1 = 300  # Default value = 300
    Transmission = Boundary_Constraint(HazeImg, A, C0, C1, windowSze)  #########77

    # Refine estimate of transmission
    regularize_lambda = 0.05  # Default value = 1 --> Regularization parameter, the more this  value, the closer to the original patch wise transmission
    sigma = 0.5
    Transmission = CalTransmission(HazeImg, Transmission, regularize_lambda, sigma)  # Using contextual information
    # Perform DeHazing
    HazeCorrectedImg = Remove_haze_from_image(HazeImg, Transmission, A, 0.78)
    return HazeCorrectedImg
    # return Remove_haze_from_image(HazeImg, Transmission, A, 0.78)
# cv2.imshow('Result', dehazed_imagee())