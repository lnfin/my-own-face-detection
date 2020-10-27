import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2

np.set_printoptions(suppress=True)

model = tensorflow.keras.models.load_model('keras_model.h5')

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

import cv2
import os
cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 2
while(True):
     face_id = 12
     ret, img = cam.read()
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     faces = face_detector.detectMultiScale(gray, 1.3, 5)
     if (faces != ()):
          for (x,y,w,h) in faces:
               cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
               count += 1
               cv2.imwrite('user.' + str(face_id) + '.' + str(count) + '.jpg',gray[y:y+h,x:x+w] )

     image = Image.open('user.' + str(face_id) + '.' + str(count) + '.jpg')

     size = (224, 224)
     image = ImageOps.fit(image, size, Image.ANTIALIAS)
     image_array = np.asarray(image)

     normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

     normalized_image_array.resize(224,224,3)
     data[0] = normalized_image_array

     prediction = model.predict(data)
     print(prediction)
     if prediction[0][0] > prediction[0][1]:
          print("Леха в маске")
     else:
          print("Лёха без маски")
     cv2.imshow('image', img)
     k = cv2.waitKey(100) & 0xff
     if k == 27:
         break
     if count >= 30:
         break




