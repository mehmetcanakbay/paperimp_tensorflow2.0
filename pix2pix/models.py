import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

class ReflectiveConv2D(layers.Layer):
    def __init__(self, filters=64, kernel_size=3, strides=1, padding=1, kernel_initialization='glorot_uniform'):
        super().__init__()
        self.padding = padding
        self.conv2d = layers.Conv2D(filters, 
                      kernel_size, strides, 
                      padding='valid', kernel_initializer=kernel_initialization)
    def call(self, x):
        out = self.conv2d(x)
        out = tf.pad(out, [[0,0], [self.padding, self.padding], 
                            [self.padding, self.padding], [0,0]], mode='REFLECT')
        return out

class Conv2D_ReLU_Downsample(keras.layers.Layer):
    def __init__(self, filters, relu=True):
        super().__init__()
        self.conv1 = ReflectiveConv2D(filters, strides=1, padding=1)
        self.ds = layers.MaxPooling2D((2,2))
        self.relu = relu
    def call(self, x):
        if self.relu:
            return tf.nn.relu(self.ds(self.conv1(x)))
        else:
            return tf.nn.leaky_relu(self.ds(self.conv1(x)))

class Conv2D_ReLU_Upsample(keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = ReflectiveConv2D(filters, strides=1, padding=1)
        self.us = layers.UpSampling2D((2,2))
    def call(self, x):
        return tf.nn.relu(self.us(self.conv1(x)))

class Discriminator(keras.models.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D_ReLU_Downsample(64, relu=False)        
        self.conv2 = Conv2D_ReLU_Downsample(128, relu=False)        
        self.conv3 = Conv2D_ReLU_Downsample(256, relu=False)        
        self.conv4 = Conv2D_ReLU_Downsample(512, relu=False)        
        # self.conv5 = Conv2D_ReLU_Downsample(512)      
        self.fc1 = layers.Dense(1, activation='sigmoid')
        self.flat = layers.Flatten()
        
    def call(self, x):
        out = tf.nn.dropout(self.conv1(x), 0.4)
        out = tf.nn.dropout(self.conv2(out), 0.4)
        out = tf.nn.dropout(self.conv3(out), 0.4)
        out = tf.nn.dropout(self.conv4(out), 0.4)
        # out = self.conv5(out)
        out = tf.nn.dropout(self.flat(out), 0.4)
        out = self.fc1(out)
        return out


class Generator(keras.models.Model):
    def __init__(self):
        super().__init__()
        #224
        self.conv1 = Conv2D_ReLU_Downsample(64) #112 
        self.conv2 = Conv2D_ReLU_Downsample(128) # 56 
        self.conv3 = Conv2D_ReLU_Downsample(256) # 28 
        self.conv4 = Conv2D_ReLU_Downsample(512) #14 
        self.conv5 = Conv2D_ReLU_Downsample(512) # 7

        self.bnconv = ReflectiveConv2D(64, 3, strides=1, padding=1)

        self.deconv1 = Conv2D_ReLU_Upsample(512) # 14
        self.deconv2 = Conv2D_ReLU_Upsample(512) # 28
        self.deconv3 = Conv2D_ReLU_Upsample(256) # 56
        self.deconv4 = Conv2D_ReLU_Upsample(128) # 112
        self.deconv5 = Conv2D_ReLU_Upsample(64) # 224

        self.out = ReflectiveConv2D(3, 1, 1, padding=0)
    
    def call(self, x):
        out1 = self.conv1(x) #64
        out2 = self.conv2(out1) # 128
        out3 = self.conv3(out2) # 256
        out4 = self.conv4(out3) # 512
        out5 = self.conv5(out4) # 512

        out = self.bnconv(out5)

        out = tf.concat([out, out5], axis=-1) # 
        out = self.deconv1(out)
        out = tf.concat([out, out4],axis=-1)
        out = self.deconv2(out)
        out = tf.concat([out, out3], axis=-1)
        out = self.deconv3(out)
        out = tf.concat([out, out2], axis=-1)
        out = self.deconv4(out)
        out = tf.concat([out, out1], axis=-1)
        out = self.deconv5(out)

        return self.out(out)
