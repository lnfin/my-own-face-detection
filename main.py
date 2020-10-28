import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2
import req
import time

id_of_camera = input("Enter id of camera ==> ")
np.set_printoptions(suppress=True)

model = tensorflow.keras.models.load_model('keras_model.h5')

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

ky = 0
kn = 0
flag = True
length = 0
start = False
while (flag):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    cv2.imshow('image', img)
    start = True
    if length < len(faces):
        for i in range(length, len(faces)):
            for (x, y, w, h) in [faces[i]]:
                cv2.rectangle(img, (x - 50, y - 50), (x + w + 50, y + h + 50), (255, 0, 0), 2)
                cv2.imwrite('user.12.2.jpg', gray[y:y + h, x:x + w])

                image = Image.open('user.12.2.jpg')

                size = (224, 224)
                image = ImageOps.fit(image, size, Image.ANTIALIAS)
                image_array = np.asarray(image)

                normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

                normalized_image_array.resize(224, 224, 3)
                data[0] = normalized_image_array

                prediction = model.predict(data)
                print(prediction)

                if (prediction[0][0] > prediction[0][1]):
                    req.send_request(id_of_camera, 1)
                else:
                    req.send_request(id_of_camera, 0)

                cv2.imshow('image', img)
                k = cv2.waitKey(100) & 0xff
                if k == 27:
                    break

    length = len(faces) if start else 0
    print(length)