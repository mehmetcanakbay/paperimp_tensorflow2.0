import cv2
import tensorflow as tf
import os 
import random
import numpy as np

def get_high_low_res():
    output_imgs = []
    input_imgs = []
    target_width, target_height = 128, 128
    for input_name in os.listdir("text_dataset/with_text"):
        print(input_name)
        input_img = cv2.imread(os.path.join("text_dataset/with_text", input_name))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        output_img = cv2.imread(os.path.join("text_dataset/neutral", input_name))
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        width, height, channels = input_img.shape

        for _ in range(32):
            width_random = random.randint(0, width-target_width)
            height_random = random.randint(0, height-target_height)
            img_cropped_input = input_img[width_random:width_random+target_width, height_random:height_random+target_height]
            img_cropped_output = output_img[width_random:width_random+target_width, height_random:height_random+target_height]
            output_imgs.append(((img_cropped_output - 127.5) / 127.5) + 1) 
            input_imgs.append(((img_cropped_input - 127.5) / 127.5) + 1) 
            
    y = tf.convert_to_tensor(output_imgs, dtype=tf.float32)
    x = tf.convert_to_tensor(input_imgs, dtype=tf.float32)
    return y, x

def get_iterable(batch_size=4):
    y, x = get_high_low_res()
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    iterable = datagen.flow(x, y, batch_size=batch_size, seed=1, shuffle=True)
    return iterable

def preprocessimg(img, channels):
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = ((img - 127.5) / 127.5) + 1
    img = np.reshape(img, (256, 256, channels))
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img

def get_test_image():
    img = cv2.imread("test_img_depre2.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessimg(img, 3)
    return img

# heya = get_iterable()
# for x, y in heya:
#     img = (((x[0]- 1) *127.5) + 127.5).astype("uint8")
#     img_y = (((y[0]- 1) *127.5) + 127.5).astype("uint8")
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     img = cv2.imwrite("hey.jpg", img)
#     img = cv2.cvtColor(img_y, cv2.COLOR_RGB2BGR)
#     img = cv2.imwrite("hey_y.jpg", img)
#     img = (((x[1]- 1) *127.5) + 127.5).astype("uint8")
#     img_y = (((y[1]- 1) *127.5) + 127.5).astype("uint8")
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     img = cv2.imwrite("heyy.jpg", img)
#     img = cv2.cvtColor(img_y, cv2.COLOR_RGB2BGR)
#     img = cv2.imwrite("hey_yy.jpg", img)
#     break