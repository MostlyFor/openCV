from PyQt5.QtWidgets import *

#Qpixmap
from PyQt5.QtGui import *


#Qt 이용하기 위해서
#from PyQt5.QtCore import *
import sys
import threading
import cv2


video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FPS,1000)

#for face detection
#faceCascade = cv2.CascadeClassifier(cv2.__path__[0]+'\data\haarcascade_frontalface_default.xml')

faceCascade = cv2.CascadeClassifier(cv2.__path__[0]+'\data\haarcascade_frontalface_alt2.xml')


def cam(img):
    while True:
        ret, frame = video.read()
        #480, 640, 3 ( , , channels)
        #rows,cols,_=frame.shape
        #print(rows,cols,_)
        
        frame_gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale(frame_gray,flags=cv2.CASCADE_SCALE_IMAGE)

        for (x,y,w,h) in faces:
            center = (x + w//2, y + h//2)
            frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        
        

        img.setPixmap(QPixmap('./pic/now.jpg')) #카메라화면 표시
        cv2.imwrite('./pic/now.jpg',frame)
        
    



class MyApp(QWidget):
    def __init__(self):
        #print('흐름도')
        super().__init__()
        self.page()
        #print(id(self))
        
    
    
    
    def AddPerson(self,e):
        global train
        train = 1
        print(train)
    
    
    def page(self):
        img = QLabel(self)
        img.move(50,0)
        img.resize(640, 480)
        
        #img.setPixmap(QPixmap('이미지 주소'))
        #img.setPixmap(QPixmap('./pic/now.jpg'))
        
        AddButton = QPushButton(self)  # 등록버튼
        
        #따라가기 위해 만든걸수도 있음.
        #AddButton.resize(numX,numY)
        AddButton.move(480, 320)
        AddButton.setText("add")
        AddButton.mousePressEvent = self.AddPerson
        
        
        #sub = threading.Thread(target = cam, args=)
        sub = threading.Thread(target = cam, args=[img])
        sub.daemon = True
        sub.start()
        
        
        
        #self.showFullScreen()
        
        self.show()

app = QApplication(sys.argv)

screen = MyApp()

screen.resize(800,480)
screen.move(0,0)

#app.exec_()는 닫기 버튼 누르면 종료됨. sys.exit()는 시스템을 종료시킴
#print('여기 먼저 실행됨.')

sys.exit(app.exec_())