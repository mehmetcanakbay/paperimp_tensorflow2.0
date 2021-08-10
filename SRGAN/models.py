#LR images by downsampling factor "r" and Gaussian filter
#Pixel shuffle = tf.nn.depth_to_space,
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import PReLU, Add, LeakyReLU, Dense
from tensorflow.keras.applications.vgg19 import preprocess_input

class UpsampleBlock(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(256, 3, 1, padding='same')
        # self.conv1 = Conv2D(128, 3, 1, padding='same')
    def call(self, x):
        out = self.conv1(x)
        out = tf.nn.depth_to_space(out, 2, data_format="NHWC")
        out = tf.nn.leaky_relu(out)
        return out

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(64, 3, 1, padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(64, 3, 1, padding='same')
        self.bn2 = BatchNormalization()
    def call(self, x):
        res = self.bn1(self.conv1(x))
        res = tf.nn.leaky_relu(res)
        res = self.bn2(self.conv2(res))
        output = Add()([res, x])
        return output

class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(64, 9, 1, padding='same')
        # self.conv1 = Conv2D(32, 9, 1, padding='same')
        self.prelu1 = PReLU()
        self.block1 = ResidualBlock()
        self.block2 = ResidualBlock()
        self.block3 = ResidualBlock()
        self.block4 = ResidualBlock()
        # self.block5 = ResidualBlock()
        # self.block6 = ResidualBlock()
        # self.block7 = ResidualBlock()
        # self.block8 = ResidualBlock()
        # self.block9 = ResidualBlock()
        # self.block10 = ResidualBlock()
        self.conv2 = Conv2D(64, 3, 1, padding='same')
        # self.conv2 = Conv2D(32, 3, 1, padding='same')
        self.bn1 = BatchNormalization()
        self.upsample1 = UpsampleBlock()
        self.upsample2 = UpsampleBlock()
        self.outputconv = Conv2D(3, 9, 1, padding='same')

    def call(self, x):
        res = tf.nn.leaky_relu(self.conv1(x))
        out = self.block1(res)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        # out = self.block5(out)
        # out = self.block6(out)
        # out = self.block7(out)
        # out = self.block8(out)
        # out = self.block9(out)
        # out = self.block10(out)
        out = self.bn1(self.conv2(out))
        out = Add()([out, res])
        out = self.upsample1(out)
        out = self.upsample2(out)
        output = tf.nn.tanh(self.outputconv(out))
        return output

class DiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, filter_size = 64, stride=1):
        super().__init__()
        self.conv = Conv2D(filter_size, 3, stride, padding='same')
        self.bn = BatchNormalization()
    
    def call(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = tf.nn.leaky_relu(out, 0.2)
        return out 

class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(32, 3, 1, padding='same')
        self.block1 = DiscriminatorBlock(64)
        self.block2 = DiscriminatorBlock(64, 2)
        self.block3 = DiscriminatorBlock(128,2)
        self.block4 = DiscriminatorBlock(64,2)
        self.block5 = DiscriminatorBlock(64,2)
        # self.block6 = DiscriminatorBlock(64)
        # self.block7 = DiscriminatorBlock(64,2)
        self.fc1 = Dense(1024)
        self.fc2 = Dense(1)

    def call(self, x):
        out = tf.nn.leaky_relu(self.conv1(x), 0.2)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        # out = self.block6(out)
        # out = self.block7(out)
        out = tf.keras.layers.Flatten()(out)
        out = tf.nn.leaky_relu(self.fc1(out), 0.2)
        output = tf.nn.sigmoid(self.fc2(out))
        return output

