import tensorflow as tf
from tensorflow.keras import initializers, layers

#Reason for this is because I also want to try out
#SSIM as a loss function. SSIM doesnt accept negative values
def custom_act(x):
    return tf.nn.tanh(x) + 1

#Gaussian distribution with mean 0 and std of 0.02
def initial(shape, dtype=None):
    init = tf.keras.initializers.RandomNormal(mean=0, stddev=0.02)
    return init(shape=shape)

class ConvBlockDown(tf.keras.layers.Layer):
    def __init__(self, channels=64, batch_norm=True):
        super().__init__()
        self.batch_norm = batch_norm
        self.conv = layers.Conv2D(channels, 4, strides=2, padding='same', kernel_initializer=initial)
        if batch_norm:
            self.bn = layers.BatchNormalization()

    def call(self, x):
        out = self.conv(x)
        if self.batch_norm:
            out = self.bn(out)
        return out

class ConvBlockUp(tf.keras.layers.Layer):
    def __init__(self, channels=64, batch_norm=True, dropout=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.conv = layers.Conv2DTranspose(channels, 4, strides=2, padding='same', kernel_initializer=initial)
        if batch_norm:
            self.bn = layers.BatchNormalization()
        if dropout:
            self.dropout = layers.Dropout(0.5)

    def call(self, x):
        out = self.conv(x)
        if self.batch_norm:
            out = self.bn(out)
        if self.dropout:
            out = self.dropout(out)
        return out

class Generator(tf.keras.models.Model):

    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlockDown(64, batch_norm=False) # 64
        self.conv2 = ConvBlockDown(128) # 32
        self.conv3 = ConvBlockDown(256) # 16
        self.conv4 = ConvBlockDown(512) # 8
        self.conv5 = ConvBlockDown(512) # 4
        self.conv6 = ConvBlockDown(512) # 2
        self.bottleneck = ConvBlockDown(512) # 1
        self.deconv1 = ConvBlockUp(512, dropout=True) # 2
        self.deconv2 = ConvBlockUp(512, dropout=True) # 4
        self.deconv3 = ConvBlockUp(512, dropout=True) # 8
        self.deconv4 = ConvBlockUp(256) # 16
        self.deconv5 = ConvBlockUp(128) # 32
        self.deconv6 = ConvBlockUp(64) # 64
        self.outputconv = layers.Conv2DTranspose(3, 4, 2, padding='same', kernel_initializer=initial)

    def call(self, x):
        res1 = tf.nn.leaky_relu(self.conv1(x))
        res2 = tf.nn.leaky_relu(self.conv2(res1))
        res3 = tf.nn.leaky_relu(self.conv3(res2))
        res4 = tf.nn.leaky_relu(self.conv4(res3))
        res5 = tf.nn.leaky_relu(self.conv5(res4))
        res6 = tf.nn.leaky_relu(self.conv6(res5))
        bn = self.bottleneck(res6)
        out = tf.nn.relu(self.deconv1(bn))
        out = tf.concat([out, res6], axis=-1)
        out = tf.nn.relu(self.deconv2(out))
        out = tf.concat([out, res5], axis=-1)
        out = tf.nn.relu(self.deconv3(out))
        out = tf.concat([out, res4], axis=-1)
        out = tf.nn.relu(self.deconv4(out))
        out = tf.concat([out, res3], axis=-1)
        out = tf.nn.relu(self.deconv5(out))
        out = tf.concat([out, res2], axis=-1)
        out = tf.nn.relu(self.deconv6(out))
        out = tf.concat([out, res1], axis=-1)
        out = self.outputconv(out)
        return custom_act(out)

class DiscConvBlock(tf.keras.layers.Layer):
    def __init__(self, channels=64, batch_norm=True):
        super().__init__()
        self.batch_norm = batch_norm
        self.conv = layers.Conv2D(channels, 4, strides=2, padding='same', kernel_initializer=initial)
        if batch_norm:
            self.bn = layers.BatchNormalization()
    def call(self, x):
        out = self.conv(x)
        if self.batch_norm:
            out = self.bn(out)
        return out

class Discriminator(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.block1 = DiscConvBlock(64, batch_norm=False)
        self.block2 = DiscConvBlock(128) #32
        self.block3 = DiscConvBlock(128) # 16
        self.block4 = DiscConvBlock(256) # 8
        self.flatten = layers.Flatten()
        self.outlayer = layers.Dense(1, activation='sigmoid')

    
    def call(self, x):
        out = tf.nn.leaky_relu(self.block1(x))
        out = tf.nn.leaky_relu(self.block2(out))
        out = tf.nn.leaky_relu(self.block3(out))
        out = tf.nn.leaky_relu(self.block4(out))
        out = self.flatten(out)
        out = self.outlayer(out)
        return out