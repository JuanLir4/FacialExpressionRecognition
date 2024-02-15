import cv2
import numpy as np

facedetectortrain = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def detectordeface(escrita):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetectortrain.detectMultiScale(gray, 1.1, 4)
    
    detected_faces = []#usado para armazenas todas as faces em uma lista
    for (x, y, z, w) in faces:
        face = gray[y:y+w, x:x+z]#retorna apenas a parte do rosto
        face = cv2.resize(face, (48, 48))
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)
        face = face / 255.0
        detected_faces.append(face)
        cv2.rectangle(frame, (x, y), (x + z, y + w), (255, 0, 0), 2)
        cv2.putText(frame, escrita, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Video", frame)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        return None  # retorna None se a tecla ESC for pressionada
    
    return detected_faces


