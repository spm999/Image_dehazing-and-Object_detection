# load yolov3 model and perform object detection
# Here we are using a pre-trained model(from yolo_algo.py) to perform object detection on an unseen photograph

import cv2
import numpy as np
from keras.models import load_model
from Estemate_airlight import Estemate_airlight
from Boundary_Constraint import Boundary_Constraint
from CalTransmission import CalTransmission
from Remove_haze_from_image import Remove_haze_from_image
from Resizing_image import Resizing_image
import numpy as np
from matplotlib.patches import Rectangle
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


HazeImg = cv2.imread('Images/image14.jpg')
# HazeImg=Resizing_image(HazeImg)
# cv2.imshow("Hazed Image",HazeImg)
## To Resize image the image, call the Resizing_image function


#Estimate Airlight
windowSze = 10
AirlightMethod = 'sss'
A = Estemate_airlight(HazeImg, AirlightMethod, windowSze)

# Calculate Boundary Constraints
windowSze = 3
C0 = 20
C1 = 300
Transmission = Boundary_Constraint(HazeImg, A, C0, C1, windowSze)  #########77

# Refine estimate of transmission
regularize_lambda = 0.05  # Default value = 1 --> Regularization parameter, the more this  value, the closer to the original patch wise transmission
sigma = 0.5
Transmission = CalTransmission(HazeImg, Transmission, regularize_lambda, sigma)  # Using contextual information

# Perform DeHazing
HazeCorrectedImg = Remove_haze_from_image(HazeImg, Transmission, A, 0.78)
cv2.imwrite('Dehazing_output_Images/The_dehazed_image.jpg', HazeCorrectedImg)
# HazeCorrectedImg=Resizing_image(HazeCorrectedImg)
# cv2.imshow("Dehazed Image",HazeCorrectedImg)

# data1 = pyplot.imread(HazeCorrectedImg)
# # plot the image
# pyplot.imshow(data1)
# fig = pyplot.figure(figsize=(10, 7))
#
# # setting values to rows and column variables
# rows = 2
# columns = 2
#
# # reading images
# Image1 = cv2.imread('Dehazing_output_Images/The_dehazed_image.jpg')
# pyplot.show()
# img1 = mpimg.imread('Dehazing_output_Images/The_dehazed_image.jpg')
# imgplot = pyplot.imshow(img1)
# pyplot.show()

# load yolov3 model
model = load_model('model.h5')

# define the expected input shape for the model
input_w, input_h = 416, 416



# load yolov3 model and perform object detection
class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]  # 0 and 1 is row and column 13*13
    nb_box = 3  # 3 anchor boxes
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))  # 13*13*3 ,-1
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if (objectness.all() <= obj_thresh): continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
            boxes.append(box)
    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


def _interval_overlap(interval_a, interval_b):########checks if two intervals overlap. Two intervals do not overlap when one ends before the other begins.
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


# intersection over union
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    # Union(A,B) = A + B - Inter(A,B)
    union = w1 * h1 + w2 * h2 - intersect
    return float(intersect) / union

# nms==Non max supression
def do_nms(boxes, nms_thresh):  # boxes from correct_yolo_boxes and  decode_netout
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


# load and prepare an image
def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename)  # load_img() Keras function to load the image .
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape)  # target_size argument to resize the image after loading
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0  # rescale the pixel values from 0-255 to 0-1 32-bit floating point values.
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height



#
photo_filename = 'Dehazing_output_Images/The_dehazed_image.jpg'
input_w, input_h = 416, 416
# load and prepare image
image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))
#print(image, image_w, image_h )
# print(image.shape)
# make prediction

# load yolov3 model
model = load_model('model.h5',compile=True)


# model = load_model('model.h5')
yhat = model.predict(image)
# summarize the shape of the list of arrays


# print([a.shape for a in yhat])
# 3 anchor boxes and 80 classes
# 13*13*3*85 (80+5)  13*13*255

# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i] * 100)

    return v_boxes, v_labels, v_scores



# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores):
    # load the image
    data = pyplot.imread(filename)

    # plot the image
    pyplot.imshow(data)
    # cv2.imshow(data)

    # get the context for drawing boxes
    ax = pyplot.gca()
    #####to get the current Axes instance on the current figure
    # ax = cv2.gca()

    # plot each box
    for i in range(len(v_boxes)):
        # by retrieving the coordinates from each bounding box and creating a Rectangle object.

        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='white')
        # draw the box
        ax.add_patch(rect)
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        pyplot.text(x1, y1, label, color='white')
        pyplot.savefig('Final/result.jpg')
    # show the plot
    pyplot.show()
    # pyplot.savefig('gh.jpg')
    # fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    # ax.plot([0, 1, 2], [10, 20, 3])
    # fig.savefig('../Output/to.png')
    # plt.savefig('foo.png')

# draw_boxes
# define the anchors
anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]

# define the probability threshold for detected objects
class_threshold = 0.5
boxes = list()
for i in range(len(yhat)):
    # decode the output of the network
    boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)

# correct the sizes of the bounding boxes for the shape of the image
correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

# suppress non-maximal boxes
do_nms(boxes, 0.5)  # Discard all boxes with pc less or equal to 0.5

# define the labels  80 labels
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# get the details of the detected objects
v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

# We can also plot our original photograph and draw the bounding box around each detected object.
for i in range(len(v_boxes)):
    print(v_labels[i], v_scores[i])
# draw what we found
fig = pyplot.figure(figsize=(10, 7))

# setting values to rows and column variables
# rows = 1
# columns = 3

# reading images
# Image1=HazeImg
# # Image2 = cv2.imread('Dehazing_output_Images/The_dehazed_image.jpg')
# Image2=HazeCorrectedImg
# Image3 = cv2.imread(photo_filename)
# fig.add_subplot(rows, columns, 1)
#
# # showing image
# plt.imshow(Image1)
# plt.axis('off')
# plt.title("First")
#
# # Adds a subplot at the 2nd position
# fig.add_subplot(rows, columns, 2)
#
# # showing image
# plt.imshow(Image2)
# plt.axis('off')
# plt.title("Second")
#
# fig.add_subplot(rows, columns, 3)
# plt.imshow(Image3)
# plt.axis('off')
# plt.title("Third")

draw_boxes(photo_filename, v_boxes, v_labels, v_scores)




# #cv2.imwrite('Output/result.jpg', result)

