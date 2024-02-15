import deeplake
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from keras.utils import to_categorical
import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


dstrain = deeplake.load('hub://activeloop/fer2013-train')
dstest = deeplake.load('hub://activeloop/fer2013-public-test')


X_train, y_train = dstrain['images'], dstrain['labels']
X_test, y_test = dstest['images'], dstest['labels']


X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)



#onehotenconder
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

#usando o facedetector para a base de dados de teste:
Xtest_frame = []
index_to_delete = 0
for img in X_test:
    faces = face_detector.detectMultiScale(img, 1.3, 4)
    
    #caso nao encontre a face, apagar a linhas do y
    if len(faces) == 0:
        y_test = np.delete(y_test, index_to_delete, axis=0)
        
    
    #x,y,w,h representam as coordenadas da face
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))
        Xtest_frame.append(face_img)
        index_to_delete = index_to_delete + 1
    
    

            
#usando o facedetector para a base de dados treinos:
Xtrain_frame = []
index_to_delete2 = 0
for img1 in X_train:
    faces1 = face_detector.detectMultiScale(img1, 1.1, 4)
    
    #caso nao encontre a face, apagar a linhas do y
    if len(faces1) == 0:
        y_train = np.delete(y_train, index_to_delete2, axis=0)
    
    #x,y,w,h representam as coordenadas da face
    for (x, y, w, h) in faces1:
        face_img = img1[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))
        Xtrain_frame.append(face_img)
        index_to_delete2 = index_to_delete2 + 1
    
    


# tranformando em arrays numpy
Xtrain_frame = np.array(Xtrain_frame)
Xtest_frame= np.array(Xtest_frame)

#expandindo dimensoes
Xtrain_frame = np.expand_dims(Xtrain_frame, axis=-1)
Xtest_frame = np.expand_dims(Xtest_frame, axis=-1)


#verificando o tamanho
# print(Xtrain_frame.shape)
# print(y_train.shape)
# print(Xtest_frame.shape)
# print(y_test.shape)



classificador = Sequential()

#aumentar numero de camadas e suas configurações
classificador.add(Conv2D(64, (3,3), input_shape = (48, 48, 1), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Conv2D(64, (3,3), input_shape = (48, 48, 1), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))


classificador.add(Flatten())


classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.3))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.3))
classificador.add(Dense(units = 7, activation = 'softmax'))


#mexer na learning hate
opt = keras.optimizers.Adam(learning_rate=0.001)
classificador.compile(optimizer = opt, loss = 'categorical_crossentropy' , metrics = ['accuracy'])

gerar_treino = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)




# # Ajustar o gerador de dados aos dados de treinamento
gerador_treino = gerar_treino.flow(Xtrain_frame, y_train, batch_size = 32)
# alidation_generator = gerar_treino.flow(Xtest_frame, y_test, batch_size=len(X_train) // 32)


# Usar fit_generator corretamente
classificador.fit_generator(gerador_treino, steps_per_epoch=100, epochs=300, validation_steps=100)

classificador.save("nn.h5")
classificador.save("neural_net")

#CRIAR A PARTE DOS TESTES

previsao = classificador.predict(Xtest_frame)
previsao_resultados = np.argmax(previsao, axis=1)

y_test_indices = np.argmax(y_test, axis=1)

contagem_erro = np.sum(previsao_resultados != y_test_indices)
print(f"de {len(Xtest_frame)} testes,  {contagem_erro} estao errados.")



