import cv2
from keras.models import load_model
import numpy as np
from facedetector import detectordeface

#carregando rede neural
new_model = load_model("nn.h5")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
resposta_escrita = None
while True:
    faces = detectordeface(resposta_escrita)  # Obter quadro normalizado
    if faces is not None:
        for face in faces:
            previsao = new_model.predict(faces)
            resposta = np.argmax(previsao)
            
            if resposta == 0:
                resposta_escrita = "raiva"
            elif resposta == 1:
                resposta_escrita = "nojo"
            elif resposta == 2:
                resposta_escrita = "medo"
            elif resposta == 3:
                resposta_escrita = "feliz"
            elif resposta == 4:
                resposta_escrita = "triste"
            elif resposta == 5:
                resposta_escrita = "surpresa"
            elif resposta == 6:
                resposta_escrita = "neutro"
            else:
                resposta_escrita = "número de classe não reconhecido"
                
            
cap.release()
cv2.destroyAllWindows()   

