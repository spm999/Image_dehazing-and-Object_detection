import cv2
def Resizing_image(HazeImg):

    Channels = cv2.split(HazeImg)
    rows, cols = Channels[0].shape
    HazeImg = cv2.resize(HazeImg, (int(0.7 * cols), int(0.7 * rows)))

    return HazeImg
