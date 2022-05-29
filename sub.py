from PyQt5.QtWidgets import *
from PyQt5.QtGui import * #Qpixmap
import sys
import threading
import cv2
import os
import numpy as np



video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FPS,1000)

train = False
person = 0

#for face detection
faceCascade = cv2.CascadeClassifier(cv2.__path__[0]+'\data\haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

path = os.path.dirname(__file__)
#recognizer.read(path+'/trainer/trainer.yml')



def face_detection(frame_gray):
    faces = faceCascade.detectMultiScale(frame_gray,flags=cv2.CASCADE_SCALE_IMAGE)
    return faces

def face_register_and_train():
    global person
    person +=1
    num_picture = 1
    images = []
    labels = []
    
    while True:
        ret, tmp_img =video.read()
        tmp_gray = cv2.cvtColor(tmp_img,cv2.COLOR_BGR2GRAY)
        faces=face_detection(tmp_gray)
        for (x,y,w,h) in faces:
            #몇 번째 사진인지만 저장
            cv2.imwrite("dataSet/face-"+'.'+ str(num_picture) + ".jpg", tmp_gray) # dataSet 폴더에 저장
            num_picture+=1
            print(num_picture,'번째 사진 저장중')
            images.append(tmp_gray[y:y+h,x:x+w])
            labels.append(person)
        if num_picture > 20:
            break
    
    recognizer.train(images, np.array(labels))
    print('저장중')
    recognizer.save('./trainer.yml')
    


def cam(img):
    while True:
        ret, frame = video.read()
        
        #사이즈 조정
        #480, 640, 3 ( , , channels)
        #rows,cols,_=frame.shape
        #print(rows,cols,_)
        
        frame_gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 검출
        faces = face_detection(frame_gray)

        # + 타원 그리기
        for (x,y,w,h) in faces:
            center = (x + w//2, y + h//2)
            frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)

            # 얼굴 인식
            if person > 0:
                recognizer.read('./trainer.yml')
                id, conf = recognizer.predict(frame_gray[y:y+h,x:x+w])
                print(conf, id,'번째 사람')
        
        
        #add 버튼 눌렀을 시 얼굴 등록 및 학습
        global train
        if train :
            face_register_and_train()
            train=False
        
        
        
        
        # 졸음 검출
        #if button == 1:
            # 졸음 검출 함수
            
        
        #화면에 이미지 전달
        cv2.imwrite('./pic/now.jpg',frame)
        img.setPixmap(QPixmap('./pic/now.jpg'))
        
    



class MyApp(QWidget):
    def __init__(self):
        #print('흐름도')
        super().__init__()
        self.page()
        #print(id(self))
        
    
    
    
    def AddPerson(self,e):
        # 이건 노트북에서만 이름 등록하게
        #self.line_edit=QLineEdit(self)
        #self.line_edit.move(100,100)
        #text=self.line_edit.text()
        #print(text)
        global train
        train = True
    
    
    def page(self):
        space = QLabel(self)
        space.move(50,0)
        space.resize(640, 480)
        
        # 등록버튼
        AddButton = QPushButton(self) 
        
        #얼굴 등록 버튼 누르면 
        AddButton.mousePressEvent = self.AddPerson
        
        #따라가기 위해 만든걸수도 있음.
        #AddButton.resize(numX,numY)
        AddButton.move(480, 320)
        AddButton.setText("add")
        
        # 공간을 인자로 넘기고 나중에 그 공간에 이미지 받음
        sub = threading.Thread(target = cam, args=[space])
        sub.daemon = True
        sub.start()
        
        
        self.show()

app = QApplication(sys.argv)

screen = MyApp()

screen.resize(800,480)
screen.move(0,0)


sys.exit(app.exec_())