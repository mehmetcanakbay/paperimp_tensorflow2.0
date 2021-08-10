import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

img_width, img_heigth = 224, 224
initialize = tf.keras.initializers.he_normal()


def load_img(PATH):
    img = cv2.imread(PATH)
    img = cv2.resize(img, (img_width, img_heigth))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.convert_to_tensor(img)
    img = tf.expand_dims(img, axis=0)
    img = tf.cast(img, tf.float32)
    return img

class Residual_Layer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(128, 
        kernel_size=3, strides=1, padding='same',
        kernel_initializer=initialize)

        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(128, 
        kernel_size=3, strides=1, padding='same',
        kernel_initializer=initialize)

        self.bn2 = BatchNormalization()

    def call(self, x):
        x_out = self.bn1(self.conv1(x))
        x_out = tf.nn.relu(x_out)
        x_out = self.bn2(self.conv2(x_out))
        output = tf.keras.layers.Add()([x_out, x])
        return output


class ImageTransformNet(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.inputconv = Conv2D(32, 
        kernel_size=9, strides=1, padding='same',
        kernel_initializer=initialize)
        self.bn1 = BatchNormalization()

        self.conv1 = Conv2D(64, 
        kernel_size=3, strides=2, padding='same',
        kernel_initializer=initialize)
        self.bn2 = BatchNormalization()

        self.conv2 = Conv2D(128, 
        kernel_size=3, strides=2, padding='same',
        kernel_initializer=initialize)
        self.bn3 = BatchNormalization()
        
        self.block1 = Residual_Layer()
        self.block2 = Residual_Layer()
        self.block3 = Residual_Layer()
        self.block4 = Residual_Layer()
        self.block5 = Residual_Layer()

        self.deconv1 = Conv2DTranspose(64, 
        kernel_size=3, strides=2, padding='same',
        kernel_initializer=initialize)
        self.bn4 = BatchNormalization()

        self.deconv2 = Conv2DTranspose(32, 
        kernel_size=3, strides=2, padding='same',
        kernel_initializer=initialize)
        self.bn5 = BatchNormalization()

        self.outputconv = Conv2D(3, kernel_size=9, strides=1, padding='same')
    
    def call(self, x):
        x = tf.nn.relu(self.bn1(self.inputconv(x)))
        x = tf.nn.relu(self.bn2(self.conv1(x)))
        x = tf.nn.relu(self.bn3(self.conv2(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = tf.nn.relu(self.bn4(self.deconv1(x)))
        x = tf.nn.relu(self.bn5(self.deconv2(x)))
        output = tf.nn.tanh(self.outputconv(x))
        return output


def get_extractors():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    conv_parts = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_part = [
        'block4_conv3'
    ]
    outputs = [vgg.get_layer(name).output for name in conv_parts]
    feature_extractor_style = tf.keras.Model([vgg.input], outputs)
    feature_extractor_content = tf.keras.Model(inputs = vgg.input, outputs= vgg.get_layer(content_part[0]).output)
    return feature_extractor_style, feature_extractor_content

def gram_matrix(extracted_tensor):
    batch, width, height, channels = extracted_tensor.shape
    vectorize = tf.reshape(extracted_tensor, [width * height, channels])
    return tf.matmul(vectorize, vectorize, transpose_a=True) / (width*height*channels)

def get_style_loss(y_hat, y_true):
    batch, width, height, channels = y_hat.shape
    y_hat_gram = gram_matrix(y_hat)
    y_true_gram = gram_matrix(y_true)
    return tf.reduce_sum(tf.square(tf.square(y_hat_gram - y_true_gram)))

def get_content_loss(y_hat, y_true):
    batch, width, height, channels = y_hat.shape
    return tf.sqrt(tf.reduce_sum(tf.square(tf.square(y_hat - y_true)))) 


net = ImageTransformNet()
content_img = load_img(r"neural_style_transfer_Johnsons\neural_style_transfer_5_0.jpg") / 127.5 - 1
style_img = load_img(r"neural_style_transfer_Johnsons\neural_style_transfer_5_1.jpg") / 127.5 - 1

feature_extractor_style, feature_extractor_content = get_extractors()
optimizer = tf.keras.optimizers.Adam(lr=1e-3)
@tf.function
def train_step(style_loss, content_loss):
    with tf.GradientTape() as tape:
        x_hat = net(content_img)
        gen_content_feat = feature_extractor_content(x_hat)
        gen_style_feat = feature_extractor_style(x_hat)
        style_feature = feature_extractor_style(style_img)
        content_feature = feature_extractor_content(content_img)
        for gen_feat, features in zip(gen_style_feat, style_feature):
            style_loss += get_style_loss(features, gen_feat) 
        content_loss = get_content_loss(content_feature, gen_content_feat)
        tv = tf.reduce_mean(tf.image.total_variation(x_hat))
        complete_loss =  content_loss + style_loss + 5e-2*tv
    grads = tape.gradient(complete_loss, net.trainable_weights)
    optimizer.apply_gradients(zip(grads, net.trainable_weights))
    return complete_loss, style_loss, content_loss

for _ in range(100):
    style_loss = 0
    content_loss = 0
    complete_loss, style_loss, content_loss = train_step(style_loss, content_loss)
    print(f"Step: {_+1}, loss {complete_loss}, Styleloss {style_loss}, content_loss {content_loss}")

print(tf.cast(tf.reshape((net(content_img) + 1) * 127.5, [224, 224, 3]), tf.uint8))
plt.imshow(tf.cast(tf.reshape((net(content_img) + 1) * 127.5, [224, 224, 3]), tf.uint8))
plt.show()
    
