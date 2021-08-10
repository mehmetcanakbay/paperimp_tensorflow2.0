import cv2
from numpy.lib.function_base import iterable
import tensorflow as tf
import os 
import random
import matplotlib.pyplot as plt
import numpy as np
def get_high_low_res():
    r = 2
    high_res = []
    low_res = []
    target_width, target_height = 128, 128

    for i in os.listdir("dataset"):
        img = cv2.imread(os.path.join("dataset", i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        width, height, channels = img.shape
        for i in range(16):

            width_random = random.randint(0, width-target_width)
            height_random = random.randint(0, height-target_height)
            img_cropped = img[width_random:width_random+target_width, height_random:height_random+target_height]
            high_res.append(img_cropped / 255) 
            img_cropped = cv2.resize(img_cropped, (target_width //r, target_height//r), interpolation=cv2.INTER_CUBIC)
            low_res.append(img_cropped / 255) 
            
    high_res = tf.convert_to_tensor(high_res, dtype=tf.float32)
    low_res = tf.convert_to_tensor(low_res, dtype=tf.float32)
        # if len(low_res) == 101:
        #     high_res = tf.convert_to_tensor(high_res, dtype=tf.float32)
        #     low_res = tf.convert_to_tensor(low_res, dtype=tf.float32)
        #     break
    print("Fin")
    return high_res, low_res

def get_iterable(batch_size=1):
    high_res, low_res = get_high_low_res()
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    iterable = datagen.flow(low_res, high_res, batch_size=16)
    # X = tf.data.Dataset.from_tensor_slices(low_res).batch(batch_size)
    # y = tf.data.Dataset.from_tensor_slices(high_res).batch(batch_size)
    return iterable

def get_test_image():
    img = cv2.imread("test_img.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (212, 212), interpolation=cv2.INTER_CUBIC)
    img = img / 255
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img



#############DEBUG
# iterabled = get_iterable(batch_size=1)
# deneme = 0
# deneme_low = 0
# print("Fin3")
# for _,(x,y) in enumerate(iterabled):

#     plt.imshow(tf.cast(tf.reshape(x[0]*255, [32,32,3]),tf.uint8))
#     plt.savefig("{0}.jpg".format("denemelow" + str(deneme_low)))
#     deneme_low+=1
    
#     plt.imshow(tf.cast(tf.reshape(y[0]*255, [128,128,3]),tf.uint8))
#     plt.savefig("{0}.jpg".format("deneme" + str(deneme)))
#     deneme+=1
    
#     if _ == 10:
#         break

