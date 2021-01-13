from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication, QPoint
from PyQt5.QtGui import QIcon, QFont
import numpy as np
import cv2, time
from threading import Thread
import math
import sys 
from itertools import combinations
import math

class ImageDetection:
    def __init__(self, frame, window_self):
        # self.label = label
        self.frame = frame
        self.window_self = window_self
        super().__init__()
        # model =  cv2.dnn.readNet('final.cfg','final_final.weights')
        # model.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        self.net = cv2.dnn_DetectionModel('final.cfg','final_final.weights')
        self.net.setInputSize(416,416)
        self.net.setInputScale(1.0 / 255)
        self.net.setInputSwapRB(True)
        self.checkClassesNames()
        self.detection();
    
    def checkClassesNames(self):
        with open('classes.names','rt') as f:
            self.names = f.read().rstrip('\n').split('\n')
    
    def detection(self):
        classes, confidences, boxes = self.net.detect(self.frame,confThreshold=0.1,nmsThreshold=0.4)
        centroid_dict = dict()
        activities = {}
        objId = 0

        #if ((classes == ()) == False)
        if(len(classes) != 0):
            for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                label = '%.2f' % confidence
                label = '%s: %s' % (self.names[classId], label)
                labelSize, baseLine = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX, 0.5,1)
                left, top, width, height = box
                if activities.get(self.names[classId]) == None:
                    activities[self.names[classId]] = 1
                else:
                    activities[self.names[classId]] += 1
                # xmin, ymin, xmax, ymax = self.convertBack(float(left),float(top),float(width),float(height))

                # cv2.rectangle(frame,box,color=(0,255,0), thickness=3)
                # cv2.rectangle(frame,(left, top-labelSize[1]),(left+labelSize[0],top+baseLine),(255,255,255),cv2.FILLED)
                # cv2.putText(frame,label,(left, top), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0))
                centroid_dict[objId] = (left, top, width, height, label)
                objId+=1
            red_zone_list = [] 
            for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
                dx, dy = ((int(p1[0])-int(p2[0]))**2), ((int(p1[1])-int(p2[1]))**2)
                distance = self.is_close(dx, dy)
                if distance < 75.0 and distance > 40.0:
                    if id1 not in red_zone_list:
                        red_zone_list.append(id1)  
                    if id2 not in red_zone_list:
                        red_zone_list.append(id2)
            for idx, box in centroid_dict.items():
                label = box[4]
                if idx in red_zone_list:
                    label = label + ' ( RISK!!! )'
                    cv2.rectangle(self.frame,(box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 2)
                else:
                    label = label + ' ( SAFE )'
                    cv2.rectangle(self.frame,(box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)

                labelSize, baseLine = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX, 0.5,1)
                cv2.rectangle(self.frame,(box[0], box[1]-labelSize[1]),(box[0]+labelSize[0],box[1]+baseLine),(255,255,255),cv2.FILLED)
                cv2.putText(self.frame,label,(box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0))

            risk = '<span style="color:#cd5d7d;">Number of People at Risk : </span><span style="color:white;">'+str(len(red_zone_list))+'</span><br>'
            safe = '<span style="color:#70af85;">Number of Safe People : </span><span style="color:white">'+ str(len(centroid_dict.items()) - len(red_zone_list))+'</span>'
            self.window_self.soc_dis.setText(risk+safe)
            _html = ''
            color= {
                'Breastfeeding' : 'red',
                'Bicycling' : 'red',
                'Cleaning' : 'red',
                'Dancing' : 'red',
                'Fishing' : 'red',
                'Running' : '#e8e9a1',
                'Walking' : '#a3ddcb',
                'Standing' : 'red',
                'Sitting' : 'red',
                'Chatting' : 'red',
                'Driving' : 'red',
                'Singing' : 'red',
                'Swimming' : 'red',
                'Washing'  : 'red',
                'Hands' : 'red',
                'Cooking' : 'red',
            }
            for key in activities:
                _html = _html + '<font color="'+color[key]+'">'+ (str(key) + ' : '+'</font><font color="white">'+  str(activities[key]))+'</font><br>'
                self.window_self.activities.setText(_html)
            # cv2.rectangle(frame,(0,0,170,20),(0,0,0),cv2.FILLED)
            # self.label.setText(label)
            # cv2.putText(frame,label,(0, 15), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))

    def is_close(self, p1, p2):
        dst = math.sqrt(p1 + p2)
        return dst 

    def convertBack(self, x, y, w, h): 
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax




class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.UI()
        self.capture = cv2.VideoCapture('Testing/vid.mp4')
        self.oldPos = self.pos()
        self.thread = Thread(target = self.vid, args=())
        self.thread.start()

    def UI(self):
        flags = QtCore.Qt.WindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowFlags(flags)
        self.setGeometry(300, 300, 300, 0)
        self.setStyleSheet("background-color: black;") 
        self.soc_dis = QLabel('<font></font>',self)
        self.soc_dis.setFont(QFont('Arial', 12)) 
        self.activities = QLabel('<font></font>',self)
        self.activities.setFont(QFont('Arial', 12)) 

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        hbox.addWidget(self.soc_dis)

        vbox.addLayout(hbox)
        vbox.addStretch()
        vbox.addWidget(self.activities)
        self.setLayout(vbox)

    def closeEvent(self):
        self.timer.cancel()

    def vid(self):
        zero = False;
        while(True):
            # Capture frame-by-frame
            ret, frame = self.capture.read()

            if ret:
                frame_sized = self.rescaleFrame(frame, .2)
                ImageDetection(frame_sized, window_self = self)
                cv2.imshow('frame',frame_sized)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if(cv2.getWindowProperty('frame', 0) == 0):
                zero = True
            if(zero and cv2.getWindowProperty('frame', 0) == -1):
                break;
        # When everything done, release the capture
        self.capture.release()
        cv2.destroyAllWindows()
        self.close()
        # self.close()
    
    def rescaleFrame(self,frame, scale=0.75):
        dimension = (600, 400)
        return cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)
    
    def mousePressEvent(self, event):
        self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        delta = QPoint (event.globalPos() - self.oldPos)
        #print(delta)
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.oldPos = event.globalPos()


    




  
if __name__=="__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
