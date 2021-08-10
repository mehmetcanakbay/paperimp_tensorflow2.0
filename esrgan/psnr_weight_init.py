import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from models import *
from dataprep import *
import time 
import matplotlib.pyplot as plt

test_img_ = get_test_image()
testbatch, testwidth, testheight, testchannels = test_img_.shape
datagen = get_iterable(batch_size=16)
optimizer = tf.keras.optimizers.Adam(lr=2e-4)
inputs = tf.keras.layers.Input(shape=(None,None,3))
model = Generator()
output = model(inputs)
generator = tf.keras.models.Model(inputs=inputs, outputs=output)
# loss_fun = tf.keras.losses.MeanAbsoluteError()
count = 0
@tf.function
def train_step(hr, lr):
    with tf.GradientTape() as tape:
        gen_hr = generator(lr)
        print(hr.shape)
        print(gen_hr.shape)
        loss = tf.reduce_sum(tf.math.abs(gen_hr - hr))
        # loss = loss_fun(hr, gen_hr)
    grads = tape.gradient(loss, generator.trainable_weights)
    optimizer.apply_gradients(zip(grads, generator.trainable_weights))
    return loss

for _ in range(200):
    print("{0} in progress".format(_))  
    for steps, (low_res, high_res) in enumerate(datagen):
        loss_ = 0        
        # print(f"Step: {steps}") 
        starttime = time.time()
        loss_ += train_step(high_res, low_res)
        end_time = time.time()
        # print("loss: " + str(loss))

        if steps>50:
            break
    print(f"LOSS: {loss_}") 
    
    # if _ == 10:
    #     optimizer.lr.assign(2e-5)
    # if _ == 20:
        # optimizer.lr.assign(5e-6) 
    if _ == 150:
        optimizer.lr.assign(2e-5)
    if _ % 2 == 0:
        plt.imshow(tf.cast(tf.reshape((generator(test_img_, training=False) *255), [212*2,212*2,3]),tf.uint8))
        plt.savefig(r"_pass_imgs/{0}.jpg".format(count))
        count += 1

# generator.save('models/model_MAE_FTH')
generator.save('tflitemodel/psnr')