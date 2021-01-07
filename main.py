from threading import Thread
import cv2, time
from itertools import combinations
import math

net = cv2.dnn_DetectionModel('yolov4.cfg','yolov4.weights')
net.setInputSize(416,416)
net.setInputScale(1 / 255)
net.setInputSwapRB(True)

with open('classes.names','rt') as f:
    names = f.read().rstrip('\n').split('\n')


def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimension = (width, height)

    return cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)

def detection(frame):
    classes, confidences, boxes = net.detect(frame,confThreshold=0.1,nmsThreshold=0.4)
    centroid_dict = dict()
    objId = 0

    #if ((classes == ()) == False)
    if(len(classes) != 0):
        for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
            label = '%.2f' % confidence
            label = '%s: %s' % (names[classId], label)
            labelSize, baseLine = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX, 0.5,1)
            left, top, width, height = box

            #xmin, ymin, xmax, ymax = convertBack(float(left),float(top),float(width),float(height))

            # cv2.rectangle(frame,box,color=(0,255,0), thickness=3)
            cv2.rectangle(frame,(left, top-labelSize[1]),(left+labelSize[0],top+baseLine),(255,255,255),cv2.FILLED)
            cv2.putText(frame,label,(left, top), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0))
            centroid_dict[objId] = (left, top, width, height)
            objId+=1

        red_zone_list = [] 
        red_line_list = []
        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
            dx, dy = ((int(p1[0])-int(p2[0]))**2), ((int(p1[1])-int(p2[1]))**2)
            distance = is_close(dx, dy)
            # print(id1,id2, distance)
            if distance < 75.0:
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)  
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)

        for idx, box in centroid_dict.items():
            if idx in red_zone_list:
                cv2.rectangle(frame, (box[0],box[1],box[2],box[3]), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (box[0],box[1],box[2],box[2]), (0, 255, 0), 3)



class ThreadedCamera(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/120
        self.FPS_MS = int(self.FPS * 1000)

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)

    def show_frame(self):
        frame = rescaleFrame(self.frame,.5)
        detection(frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(self.FPS_MS)


def vid():
    if __name__ == '__main__':
        src = 'Testing/vid.mp4'
        threaded_camera = ThreadedCamera(src)
        while True:
            try:
                threaded_camera.show_frame()
            except AttributeError:
                pass

def is_close(p1, p2):
    """
    # 1. Calculate Euclidean Distance of two points    
    :param:
    p1, p2 = two points for calculating Euclidean Distance

    :return:
    dst = Euclidean Distance between two 2d points
    """
    dst = math.sqrt(p1 + p2)
    return dst 

def convertBack(x, y, w, h): 
    """
    # 2. Converts center coordinates to rectangle coordinates
    :param:
    x, y = midpoint of bounding box
    w, h = width, height of the bounding box
    
    :return:
    xmin, ymin, xmax, ymax
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax



def im():
    image = cv2.imread('Testing/Picture3.png')
    image = rescaleFrame(image)
    detection(image)
    cv2.imshow('Image', image)
    cv2.waitKey(0)

# im()
vid()