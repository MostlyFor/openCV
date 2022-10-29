from PyQt5.QtWidgets import *
from PyQt5.QtGui import * #Qpixmap
from PyQt5 import QtCore, QtGui, QtWidgets

import sys
import threading
import cv2
import os
import numpy as np
import json

#졸음 탐지
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import RPi.GPIO as GPIO


video = cv2.VideoCapture(-1)
video.set(cv2.CAP_PROP_FPS,5)

train = False

name = "start"

#for face detection
faceCascade = cv2.CascadeClassifier("/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

#face detector와 landmark predictor
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")


personID = 0;
GPIO.setmode(GPIO.BCM)
GPIO.setup(26, GPIO.OUT)

with open('registered.json') as f:
    data = json.load(f)
    print(data)
    personID = data['personID']


def createNameFolder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    except OSError:
        print('저장된 파일이 없습니다.')


def face_detection(frame_gray):
    faces = faceCascade.detectMultiScale(frame_gray,flags=cv2.CASCADE_SCALE_IMAGE)
    return faces


def face_register_and_train():
    global personID
    personID +=1
    print(personID)
    
    tmp ={}
    with open('registered.json') as f:
        data = json.load(f)
        tmp = data    
    with open('registered.json','w') as f :
        tmp[str(personID)] = name
        tmp['personID']=personID
        json.dump(tmp,f,indent="\t")
    
    
    num_picture = 1
    images = []
    labels = []
    
    while True:
        ret, tmp_img =video.read()
        tmp_gray = cv2.cvtColor(tmp_img,cv2.COLOR_BGR2GRAY)
        faces=face_detection(tmp_gray)
        
        
        # 이름에 맞는 폴더 저장
        createNameFolder('dataSet/'+name)
        
        for (x,y,w,h) in faces:
            #개인에 해당하는 폴더에 사진 저장
            cv2.imwrite("dataSet/"+name+"/"+str(num_picture) + ".jpg", tmp_gray) # dataSet 폴더에 저장
            num_picture+=1
            images.append(tmp_gray[y:y+h,x:x+w])
            labels.append(personID)
        if num_picture > 20:
            break
    
    
    recognizer.train(images, np.array(labels))
    recognizer.save(f'./dataSet/{name}/trainer.yml')
    
    
#랜드마크 감지
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]#dlib 왼쪽눈 랜드마크 지점 36,41, 튜플. 시작점과 끝점만 저장
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]#dlib 오른쪽눈 랜드마크 지점 42,47 튜플. 시작점과 끝점만 저장

EYE_AR_THRESH = 0.25
EYE_CLOSE_TIME = 3
eyeCOUNTER = 0


# 눈 감지 input은 facecasscade 사용
def eye_detect(x,y,w,h,frame_gray):
    dlibrect = dlib.rectangle(int(x),int(y),int(x+w),int(y+h))
    Landmarks = predictor(frame_gray,dlibrect)
    Landmarks = face_utils.shape_to_np(Landmarks)
    leftEye = Landmarks[lStart:lEnd]
    rightEye = Landmarks[rStart:rEnd]
    
    return leftEye,rightEye


#눈 비율 반환해주는 함수
def eye_aspect_ratio(leftEye,rightEye):
    
    EAR = 0
    for eye in (leftEye,rightEye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])

        EAR += (A + B) / (2.0 * C)
        
    return EAR/2



def sleep_detection(frame,frame_gray,x,y,w,h):
    global eyeCOUNTER
    leftEye,rightEye = eye_detect(x,y,w,h,frame_gray)
    EAR = eye_aspect_ratio(leftEye,rightEye)
    
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
    
    #눈 비율 출력
    print("\neye : " + str(EAR))
    #print(eyeCOUNTER)
    if EAR < EYE_AR_THRESH:
        eyeCOUNTER = eyeCOUNTER + 1
            
    elif EAR >= EYE_AR_THRESH:
        print ('open Eye')
        eyeCOUNTER = 0
    
    print("eyeCOUNTER : " + str(eyeCOUNTER)+"\n")    
        
    if eyeCOUNTER >= EYE_CLOSE_TIME:
        eyeCOUNTER = 0  
        print ('close Eye')
        GPIO.output(26,1)
    
    elif eyeCOUNTER < EYE_CLOSE_TIME:
        GPIO.output(26,0)
    
    


#얼굴 인식
def face_cognition(frame,frame_gray,x,y,w,h,name_path):
    if name == 'start': return
    #존재하는 사용자면 얼굴 인식 시작
    if  os.path.isdir(name_path):
            recognizer.read(name_path+'/trainer.yml')
            personID, conf = recognizer.predict(frame_gray[y:y+h,x:x+w])
            print(personID, conf)
            person_name = ""
            with open('registered.json') as f:
                data = json.load(f)
                print(personID)
                person_name = data[str(personID)]
            
            if conf < 80 :
                #cv2.putText(frame,"FONT_HERSHEY_DUPLEX",(30,110),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255))
                print(person_name,'이 맞습니다.')
                GPIO.output(26,0)
            else :
                print(person_name,'이 아닙니다.')
                GPIO.output(26,1)
    

def cam(img):
    while True:
        ret, frame = video.read()
        frame_gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 검출
        faces = face_detection(frame_gray)

        for (x,y,w,h) in faces:
            
            #얼굴 표시
            center = (x + w//2, y + h//2)
            frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
            
            #졸음 감지
            #공부모드라면 실행
            sleep_detection(frame,frame_gray,x,y,w,h)
            
            #자리 이용자
            name_path = f'./dataSet/{name}'
            
            
            # 얼굴 인식
            face_cognition(frame,frame_gray,x,y,w,h,name_path)
            
        
        #add 버튼 눌렀을 시 얼굴 등록 및 학습
        global train
        
        if train :
            face_register_and_train()
            train=False
        
            
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
        global train
        train = True
        ui = Ui_MainWindow()
    
    
    def page(self):
        space = QLabel(self)
        space.move(50,0)
        space.resize(640, 480)
        
        # 등록버튼
        AddButton = QPushButton(self) 
        
        #얼굴 등록 버튼 누르면 
        AddButton.mousePressEvent = self.AddPerson
        
        AddButton.move(480, 320)
        AddButton.setText("add")
        
        # 공간을 인자로 넘기고 나중에 그 공간에 이미지 받음
        sub = threading.Thread(target = cam, args=[space])
        sub.daemon = True
        sub.start()
    
        self.show()

class Ui_MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi()

    def setupUi(self):
        self.setWindowTitle('LineEdit')
        self.resize(300, 300)

        self.line_edit = QLineEdit(" ",self)
        self.line_edit.move(75,75)

        self.text_label = QLabel(self)
        self.text_label.move(75, 125)
        self.text_label.setText('사용자 이름')

        self.button = QPushButton(self)
        self.button.move(75, 175)
        self.button.setText('등록')
        self.button.clicked.connect(self.button_event)

        self.show()

    def button_event(self):
        global name
        name = self.line_edit.text() # line_edit text 값 가져오기
        self.text_label.setText(name) # label에 text 설정하기
        name = name.replace(' ', '')
        print(name)


class Ui_Window2(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 200)

        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
       
        # Warning_button
        self.warning_button = QtWidgets.QPushButton(self.centralwidget)
        self.warning_button.setGeometry(QtCore.QRect(150, 90, 100, 30))
        self.warning_button.setObjectName("Noise_button")
        self.warning_button.setText("Noise")
        self.warning_button.clicked.connect(self.Warning_event)

        self.show()

    def Warning_event(self):
        buttonReply = QMessageBox.warning(
            self, 'Noise Warning', "소음이 발생했습니다. 주의해주세요."
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)

    screen = MyApp()

    screen.resize(800,480)
    screen.move(0,0)

    ui = Ui_MainWindow()
    ui2 = Ui_Window2()
    sys.exit(app.exec_())
