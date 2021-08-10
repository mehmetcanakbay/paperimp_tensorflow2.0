import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.python.keras.applications.vgg19 import preprocess_input

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# img_width, img_heigth = 800, 530

def gram_matrix(extracted_tensor):
    batch, width, heigth, channels = extracted_tensor.shape 
    vectorize = tf.reshape(extracted_tensor, [width* heigth, channels])
    return tf.matmul(vectorize, vectorize, transpose_a=True)

def load_img(PATH, img_width=800, img_heigth=530):
    img = cv2.imread(PATH)
    img = cv2.resize(img, (img_width, img_heigth))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.convert_to_tensor(img)
    img = tf.expand_dims(img, axis=0)
    img = tf.cast(img, tf.float32)
    return img

def get_extractors():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    conv_parts = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1'
    ]
    content_part = [
        'block4_conv2'
    ]
    outputs = [vgg.get_layer(name).output for name in conv_parts]
    feature_extractor_style = tf.keras.Model([vgg.input], outputs)
    feature_extractor_content = tf.keras.Model(inputs = vgg.input, outputs= vgg.get_layer(content_part[0]).output)
    return feature_extractor_style, feature_extractor_content

optimizer = tf.keras.optimizers.Adam(lr=1e-2)
content_img = load_img(r"neural_style_transfer_5_0.jpg") / 127.5 - 1
style_img = load_img(r"neural_style_transfer_5_1.jpg")  / 127.5 - 1
generated_img = tf.Variable(tf.identity(content_img), dtype=tf.float32, trainable=True)
# generated_img = tf.Variable(tf.zeros(content_img.shape), dtype=tf.float32, trainable=True)
feature_extractor_style, feature_extractor_content = get_extractors()
# w1,w2 = 1e-9,1e-4
w1,w2 = 1e-3,1
@tf.function
def train_step(style_loss, content_loss):
    with tf.GradientTape() as tape:
        tape.watch(generated_img)
        pre_vgg_gen_img = preprocess_input((generated_img * 127.5) + 127.5)
        pre_vgg_style_img = preprocess_input((style_img * 127.5) + 127.5)
        gen_img_feat = feature_extractor_style(pre_vgg_gen_img)
        style_img_feat = feature_extractor_style(pre_vgg_style_img)
        for gen_feat, style_feat in zip(
            gen_img_feat,
            style_img_feat):
            batch, width, height, channels = gen_feat.shape
            gen_gram = gram_matrix(gen_feat)
            style_gram = gram_matrix(style_feat)
            style_loss += tf.reduce_sum((gen_gram - style_gram) **2) / (4*((channels**2) * ((height*width)**2)))
            # style_loss += tf.reduce_sum((gen_gram - style_gram) **2) / (4*(channels**2)*((width*heigth)**2))
        # style_loss *= 0.2
        gen_img_feat_content = feature_extractor_content(generated_img)
        content_img_feat = feature_extractor_content(content_img)
        tv = tf.reduce_mean(tf.image.total_variation(generated_img))
        content_loss += tf.reduce_sum((gen_img_feat_content - content_img_feat) ** 2) / 2
        loss = (content_loss * w1) + (style_loss * w2) + 1e-7*tv
    grads = tape.gradient(loss, generated_img)
    # optimizer.apply_gradients([(grads), (generated_img)])
    optimizer.apply_gradients(zip([grads], [generated_img]))
    return style_loss, content_loss, loss

for step in range(4000):
    style_loss = 0
    content_loss = 0
    style_loss, content_loss, loss = train_step(style_loss, content_loss)
    
    print(f"Step {step} is over")
    print(f"Complete loss: {loss}")
    print(f"Content loss {content_loss}")
    print(f"Style loss {style_loss}")
    print(f"Effective Style Loss {w2*style_loss}")
    if step % 50 == 0:
        batch, width, height, channels = generated_img.shape
        # print((tf.reshape(generated_img, [width,height,channels]) + 1) * 127.5)
        plt.imshow(tf.cast((tf.reshape(generated_img, [width,height,channels]) +1) * 127.5, tf.uint8))
        plt.savefig(r"{0}.jpg".format(step))

