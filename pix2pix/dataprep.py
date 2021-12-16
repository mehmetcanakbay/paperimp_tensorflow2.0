import cv2
import tensorflow as tf
import os
import numpy as np


X_PATH = r"C:\Users\mehmetcan\Desktop\pix2pix-dataset_SHOE\X"
Y_PATH = r"C:\Users\mehmetcan\Desktop\pix2pix-dataset_SHOE\y"

def read_img(PATH):
    img = cv2.imread(PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    return img

def get_test_img():
    img = read_img("test_img.jpg")
    return img

def _generator():
    for img in os.listdir(X_PATH):
        X_img = read_img(os.path.join(X_PATH, img))
        y_img = read_img(os.path.join(Y_PATH, img))
        yield X_img, y_img


def get_dataset_obj():
    gen = tf.data.Dataset.from_generator(_generator, output_signature=(
        tf.TensorSpec(shape=(224,224,3), dtype=tf.float32),
        tf.TensorSpec(shape=(224,224,3), dtype=tf.float32)
    ))

    gen = gen.batch(8)
    return gen