import numpy as np  
import cv2

face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 15.0, (640,480))
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #processing image in gray format, original photo will be in color format
    faces = face_cascade.detectMultiScale(gray,1.2,3) #Number of faces detecting 3

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2) #detecting square in red color

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in eyes:
             cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (255,255,255),2)

    cv2.imshow('vid', img)
    if(cv2.waitKey(1) & 0xFF== ord('q')):
        break
    
cap.release() #capture released
out.release() #output released
cv2.destroyAllWindows() #destroying every window of cv2