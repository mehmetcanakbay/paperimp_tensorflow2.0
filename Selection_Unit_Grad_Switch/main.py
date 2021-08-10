from __future__ import generator_stop
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
from model import Net
from dataprep import *
import matplotlib.pyplot as plt

datagen = get_iterable(batch_size_=16)
test_img_ = get_test_image()
#to get past input signature problem
inputs = tf.keras.layers.Input(shape=(None,None,3))
gene = Net()
outputs = gene(inputs)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
#
optimizer = tf.keras.optimizers.Adam(lr=1e-4)
theta = 1e-4
count = 0
loss_fun = tf.keras.losses.MeanAbsoluteError()
@tf.function
def train_step(lr, hr):
    with tf.GradientTape() as tape:
        gen_hr = model(lr)
        loss = loss_fun(hr, gen_hr)
    grads = tape.gradient(loss, model.trainable_weights)
    #Gradient "switch"
    grads_clipped = [(tf.where(grad < 0, -theta, theta)) for grad in grads]
    optimizer.apply_gradients(zip(grads_clipped, model.trainable_weights))
    return loss

for epoch in range(100):
    for step, (lr, hr) in enumerate(datagen):
        loss = train_step(lr, hr)
        print(f"Epoch: {epoch}")
        print(f"Current step: {step}")
        print(f"Loss: {loss}")
    #Pixel loss
    #tf.reduce_sum(tf.square(y_true - y_pred)) / w*h*c
        if step>50:
            break

    if epoch % 2 == 0:
        output = model(test_img_, training=False)[0]
        print(output.shape)
        plt.imshow(tf.cast(tf.reshape(output *255, [212*2,212*2,3]),tf.uint8))
        plt.savefig("{0}.jpg".format(count))
        count += 1

model.save("model/model1")