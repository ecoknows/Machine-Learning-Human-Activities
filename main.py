from threading import Thread
import cv2, time
from itertools import combinations
import math

net = cv2.dnn_DetectionModel('final.cfg','final.weights')
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
            # labelSize, baseLine = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX, 0.5,1)
            left, top, width, height = box

            xmin, ymin, xmax, ymax = convertBack(float(left),float(top),float(width),float(height))

            # cv2.rectangle(frame,box,color=(0,255,0), thickness=3)
            # cv2.rectangle(frame,(left, top-labelSize[1]),(left+labelSize[0],top+baseLine),(255,255,255),cv2.FILLED)
            # cv2.putText(frame,label,(left, top), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0))
            centroid_dict[objId] = (left, top, width, height,label)
            objId+=1

        red_zone_list = [] 
        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
            dx, dy = ((int(p1[0])-int(p2[0]))**2), ((int(p1[1])-int(p2[1]))**2)
            distance = is_close(dx, dy)
            #print(id1,id2, distance)
            if distance < 75.0 and distance > 40.0:
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)  
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)

        for idx, box in centroid_dict.items():
            label = box[4]
            if idx in red_zone_list:
                label = box[4] +  ' [ At Risk ]'
                labelSize, baseLine = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX, 0.5,1)
                cv2.rectangle(frame, (box[0],box[1],box[2],box[3]), (0, 0, 255), 2)
                cv2.rectangle(frame,(box[0], box[1]-labelSize[1]),(box[0]+labelSize[0],box[1]+baseLine),(255,255,255),cv2.FILLED)
                cv2.putText(frame,label,(box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0))
            else:
                label = box[4] +  ' [ Safe ]'
                labelSize, baseLine = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX, 0.5,1)
                cv2.rectangle(frame, (box[0],box[1],box[2],box[3]), (0, 255, 0), 2)
                cv2.rectangle(frame,(box[0], box[1]-labelSize[1]),(box[0]+labelSize[0],box[1]+baseLine),(255,255,255),cv2.FILLED)
                cv2.putText(frame,label,(box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0))

            labelSize, baseLine = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX, 0.5,1)
            cv2.rectangle(frame,(100, 100),(100,50),(255,255,255),cv2.FILLED)
            cv2.rectangle(frame,(box[0], box[1]-labelSize[1]),(box[0]+labelSize[0],box[1]+baseLine),(255,255,255),cv2.FILLED)
            cv2.putText(frame,label,(box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0))
        label = "People at Risk : " + str(len(red_zone_list))
        cv2.rectangle(frame,(0,0,170,20),(0,0,0),cv2.FILLED)
        cv2.putText(frame,label,(0, 15), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))


class ThreadedCamera(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        zero = False;
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)
            if(cv2.getWindowProperty('frame', 0) == 0):
                zero = True
            if(zero and cv2.getWindowProperty('frame', 0) == -1):
                break;

    def show_frame(self):
        frame = rescaleFrame(self.frame, .7)
        detection(frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(self.FPS_MS)


def vid():
    if __name__ == '__main__':
        src = 'Testing/video1.mp4'
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
    image = rescaleFrame(image,1.5)
    detection(image)
    cv2.imshow('Image', image)
    cv2.waitKey(0)

# im()
vid()