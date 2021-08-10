import cv2
import numpy as np
import tensorflow as tf
import os 
import matplotlib.pyplot as plt

def get_high_low_res():
    r = 4
    high_res = []
    low_res = []
    width, height = 384, 384

    for i in os.listdir("dataset"):
        img = cv2.imread(os.path.join("dataset", i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, height))
        high_res.append((img -127.5) / 127.5) 
        img = cv2.GaussianBlur(img, (11,11),0)
        img = cv2.resize(img, (width //4, height//4))
        low_res.append(img / 255)
        if len(low_res) == 101:
            high_res = tf.convert_to_tensor(high_res, dtype=tf.float32)
            low_res = tf.convert_to_tensor(low_res, dtype=tf.float32)
            break
    return high_res, low_res

def get_only_one_img():
    r = 4
    high_res = []
    low_res = []
    width, height = 384, 384
    # img = cv2.imread(os.path.join("cars_test", "00032.jpg"))
    img = cv2.imread(os.path.join("kedi", "kedifotografi.jpg"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))
    high_res.append((img -127.5) / 127.5) 
    img = cv2.GaussianBlur(img, (11,11),0)
    img = cv2.resize(img, (width //4, height//4))
    low_res.append(img / 255)
    high_res = tf.convert_to_tensor(high_res, dtype=tf.float32)
    low_res = tf.convert_to_tensor(low_res, dtype=tf.float32)
    return high_res, low_res

def get_iterable(batch_size=1):
    high_res, low_res = get_high_low_res()
    X = tf.data.Dataset.from_tensor_slices(low_res).batch(batch_size)
    y = tf.data.Dataset.from_tensor_slices(high_res).batch(batch_size)
    return X, y

