# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 20:48:35 2018

@author: delou
"""

import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizers/face-trainner.yml")
labels = {}
with open("pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    #Inverte a chave pelo nome
    labels = {v:k for k,v in og_labels.items()}
    
    

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.6, minNeighbors=5)
    
    for(x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #(y:cord start, ycordEnd)
        roi_color = frame[y:y+h, x:x+w]
        img_item = "my_image.png"
        cv2.imwrite(img_item, roi_gray)
        
        #Aqui podemos usar deeplearn para reconhecimento facial keras, tensorflow pytorsh, scikit learn
        id_, conf = recognizer.predict(roi_gray)
        print("Confianca: " + str(conf))
        color = ()
    
        if conf >= 55 and conf <=101:
            print(id_)
            print(labels[id_])
            
            confianca = int(conf)
            
            font=cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_] + " " + str(confianca)
            color = (255,128,0)
            stroke = 1
            cv2.putText(frame, name,(x,y),font,1,color,stroke,cv2.LINE_AA)
        else:
            print(id_)
            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (0,0,225)
            stroke = 1
            cv2.putText(frame, "Desconhecido",(x,y),font,1,color,stroke,cv2.LINE_AA)
        #Desenhar um retangulo
        stroke = 2
        end_cord_x = x +y
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke )
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    
    
cap.release()
cv2.destroyAllWindows()