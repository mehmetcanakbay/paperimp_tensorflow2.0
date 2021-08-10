import tensorflow as tf


class SEL_Unit(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(64, kernel_size=(1,1), 
                                            strides=(1,1), padding='same')
    def call(self, x):
        output = tf.nn.sigmoid(self.conv(tf.nn.relu(x)))
        return output + x

class Conv_Res_Layer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(3,3), 
                                            strides=(1,1), padding='same')
                                            
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3,3), 
                                            strides=(1,1), padding='same')

        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=(3,3), 
                                            strides=(1,1), padding='same')


        self.sel1 = SEL_Unit()
        self.sel2 = SEL_Unit()
        self.sel3 = SEL_Unit()

    def call(self, x):
        res = self.conv1(x)
        output = self.sel2(self.conv2(self.sel1(res)))
        output = self.conv3(output)
        output = output + res
        return self.sel3(output)

class Net(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.res1 = Conv_Res_Layer()
        self.res2 = Conv_Res_Layer()
        self.res3 = Conv_Res_Layer()
        self.res4 = Conv_Res_Layer()
        self.res5 = Conv_Res_Layer()
        self.res6 = Conv_Res_Layer()
        self.res7 = Conv_Res_Layer()
        self.conv1 = tf.keras.layers.Conv2D(48, kernel_size=(3,3), 
                                        strides=(1,1), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(32, kernel_size=(3,3), 
                                        strides=(1,1), padding='same')
        self.conv3 = tf.keras.layers.Conv2D(3, kernel_size=(3,3), 
                                        strides=(1,1), padding='same')
        #Nearest instead of bicubic
        self.upsample = tf.keras.layers.UpSampling2D((2,2), interpolation='nearest')

    def call(self, x):
        out = self.res1(x)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.res6(out)
        out = self.res7(out)
        out = self.conv1(out)
        # out = out + x
        out = self.upsample(self.conv2(out))
        output = self.conv3(out)
        output = output + self.upsample(x)
        return tf.nn.sigmoid(output)

