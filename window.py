from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from itertools import combinations
from PyQt5.QtGui import QPixmap
from PyQt5 import QtGui
import numpy as np
import cv2, time
import math
import sys 


class ImageDetection:
    def __init__(self, frame):
        # self.label = label
        self.net = cv2.dnn_DetectionModel('final.cfg','final.weights')
        self.net.setInputSize(416,416)
        self.net.setInputScale(1.0 / 255)
        self.net.setInputSwapRB(True)
        self.checkClassesNames()
        self.detection(frame);
    
    def checkClassesNames(self):
        with open('classes.names','rt') as f:
            self.names = f.read().rstrip('\n').split('\n')
    
    def detection(self,frame):
        classes, confidences, boxes = self.net.detect(frame,confThreshold=0.1,nmsThreshold=0.4)
        centroid_dict = dict()
        objId = 0

        #if ((classes == ()) == False)
        if(len(classes) != 0):
            for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                label = '%.2f' % confidence
                label = '%s: %s' % (self.names[classId], label)
                labelSize, baseLine = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX, 0.5,1)
                left, top, width, height = box

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
                if distance < 75.0:
                    if id1 not in red_zone_list:
                        red_zone_list.append(id1)  
                    if id2 not in red_zone_list:
                        red_zone_list.append(id2)
            for idx, box in centroid_dict.items():
                label = box[4]
                if idx in red_zone_list:
                    label = label + ' ( RISK!!! )'
                    cv2.rectangle(frame,(box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 3)
                else:
                    label = label + ' ( SAFE )'
                    cv2.rectangle(frame,(box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 3)

                labelSize, baseLine = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX, 1,1)
                cv2.rectangle(frame,(100, 100),(100,50),(255,255,255),cv2.FILLED)
                cv2.rectangle(frame,(box[0], box[1]-labelSize[1]),(box[0]+labelSize[0],box[1]+baseLine),(255,255,255),cv2.FILLED)
                cv2.putText(frame,label,(box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0), thickness=1)
            # label = "People at Risk : " + str(len(red_zone_list))
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



class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
            
    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture('Testing/vid.mp4')
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                ImageDetection(cv_img)
                self.change_pixmap_signal.emit(cv_img)
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt live label demo")
        self.disply_width = 1000
        self.display_height = 1000
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.textLabel = QLabel('Webcam')

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()
        
    def closeEvent(self, event):
        self.thread.stop()
        event.accept()



    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(416, 416, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
