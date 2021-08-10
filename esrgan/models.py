import tensorflow as tf
from tensorflow import keras
#Network interpolation
#Perceptual Loss
#PSNR
#MATLAB bicubic
#---
#Initialization should be MSRA * 0.1 +
residual_scaling_parameters_beta = 0.2

def scaled_HeNormal_initializer(shape, dtype=None):  
    initializer = tf.keras.initializers.HeNormal()
    value = initializer(shape=shape)
    return value * 0.1

class UpsampleBlock(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.upsampler = keras.layers.UpSampling2D((2,2), interpolation='nearest')
        # self.upsampler2 = keras.layers.UpSampling2D((2,2), interpolation='nearest')
        self.conv1 = keras.layers.Conv2D(48, 3, 1, padding='same', kernel_initializer=scaled_HeNormal_initializer)
        # self.conv2 = keras.layers.Conv2D(128, 3, 1, padding='same', kernel_initializer=scaled_HeNormal_initializer)

    def call(self, x):
        out = tf.nn.leaky_relu(self.conv1(self.upsampler(x)))
        # out = tf.nn.leaky_relu(self.conv2(self.upsampler2(out)))
        return out


class RRDB_DenseBlock(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=(1,1),
            kernel_initializer=scaled_HeNormal_initializer,
            padding='same'         
        )
        self.conv2 = keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=(1,1),
            kernel_initializer=scaled_HeNormal_initializer ,
            padding='same'              
        )
        self.conv3 = keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=(1,1),
            kernel_initializer=scaled_HeNormal_initializer,
            padding='same'             
        )
        self.conv4 = keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=(1,1),
            kernel_initializer=scaled_HeNormal_initializer ,
            padding='same'              
        )
        self.conv5 = keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=(1,1),
            kernel_initializer=scaled_HeNormal_initializer,
            padding='same'         
        )
    def call(self, x):
        x1 = tf.nn.leaky_relu(self.conv1(x))
        x1 = tf.add(x1,x)
        x2 = tf.nn.leaky_relu(self.conv2(x1))
        x2 = tf.add_n([x2,x1,x])
        x3 = tf.nn.leaky_relu(self.conv3(x2))
        x3 = tf.add_n([x3,x2,x1,x])
        x4 = tf.nn.leaky_relu(self.conv4(x3))
        x4 = tf.add_n([x4, x3, x2, x1, x])

        outputs = self.conv5(x4)
        return outputs * residual_scaling_parameters_beta + x


class RRDB(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.block1 = RRDB_DenseBlock()
        self.block2 = RRDB_DenseBlock()
        self.block3 = RRDB_DenseBlock()
    def call(self, x):
        forward = self.block1(x) 
        forward = self.block2(forward) 
        forward = self.block3(forward)
        return tf.add(forward * residual_scaling_parameters_beta, x)

class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer=scaled_HeNormal_initializer,
            padding='same'         
        )
        self.block1 = RRDB()
        self.block2 = RRDB()
        self.block3 = RRDB()
        # self.block4 = RRDB()
        # self.block5 = RRDB()
        # self.block6 = RRDB()
        # self.block7 = RRDB()
        # self.block8 = RRDB()
        # self.block9 = RRDB()
        # self.block10 = RRDB()

        self.conv2 = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer=scaled_HeNormal_initializer,
            padding='same'         
        )
        #
        self.lastconv1 = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer=scaled_HeNormal_initializer,
            padding='same'         
        )
        self.lastconv2 = keras.layers.Conv2D(
            filters=3,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer=scaled_HeNormal_initializer,
            padding='same'         
        )

        self.upsample = UpsampleBlock()

    def call(self, x):  
        firstpass = self.conv1(x)
        out = self.block1(firstpass)
        out = self.block2(out)
        out = self.block3(out)
        # out = self.block4(out)
        # out = self.block5(out)
        # out = self.block6(out)
        # out = self.block7(out) 
        # out = self.block8(out)
        # out = self.block9(out)
        # out = self.block10(out)
        out = self.conv2(out)
        out = tf.add_n([out, firstpass])
        out = self.upsample(out)
        out = tf.nn.leaky_relu(self.lastconv1(out))
        out = tf.nn.sigmoid(self.lastconv2(out))
        # out = self.lastconv2(out)

        return out
        
class DiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, filter_size = 32, stride=1):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filter_size, 3, stride, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
    
    def call(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = tf.nn.leaky_relu(out)
        return out 

class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, 1, padding='same')
        self.block1 = DiscriminatorBlock(32)
        self.block2 = DiscriminatorBlock(32,2)
        self.block3 = DiscriminatorBlock(32,2)
        self.block4 = DiscriminatorBlock(32,2)
        # self.block5 = DiscriminatorBlock(128,2)
        self.fc1 = tf.keras.layers.Dense(1024)
        self.fc2 = tf.keras.layers.Dense(1)

    def call(self, x):
        out = tf.nn.leaky_relu(self.conv1(x), 0.2)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        # out = self.block5(out)
        out = tf.keras.layers.GlobalAveragePooling2D()(out)
        out = tf.nn.leaky_relu(self.fc1(out), 0.2)
        output = self.fc2(out)
        return output
