import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("tflitemodel/interpolated")

img = cv2.imread(r"C:\Users\shunnie\Desktop\mordor.jpg")
# img = cv2.imread("test_img.jpg")
width, height, channels = img.shape
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (height//1, width//1))
img = img/255
img = tf.convert_to_tensor(img)
img = tf.expand_dims(img, axis=0)
hr = model.predict(img)[0]
hr = hr*255
hr = tf.cast(hr, tf.uint8)
hr = hr.numpy()
hr = cv2.cvtColor(hr, cv2.COLOR_RGB2BGR)
cv2.imshow('oldumu', hr)
cv2.imwrite("deneme.jpg", hr)
cv2.waitKey(0)
hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
plt.imshow(hr)
plt.show()

