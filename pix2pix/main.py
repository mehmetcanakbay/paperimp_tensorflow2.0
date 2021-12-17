from models import *
from dataprep import get_dataset_obj, get_test_img
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import cv2

count = 0
test_img_ = get_test_img()
gen = Generator()
disc = Discriminator()
gen_opt = keras.optimizers.Adam(lr=2e-4, beta_1=0.5, beta_2=0.999)
disc_opt = keras.optimizers.Adam(lr=2e-4, beta_1=0.5, beta_2=0.999)

dataset = get_dataset_obj()
loss_fn = keras.losses.BinaryCrossentropy()

def discriminator_loss(img, fake=False):
    losses = []
    patches_fake = tf.image.extract_patches(img[0], [1,56,56,1], [1,56,56,1], [1,1,1,1], 'VALID')
    patches_real = tf.image.extract_patches(img[1], [1,56,56,1], [1,56,56,1], [1,1,1,1], 'VALID')
    for c in range(patches_fake.shape[1]):
        for j in range(patches_fake.shape[2]):
            real_img = tf.reshape(patches_real[:,c,j,:], [-1,56,56,3])
            fake_img = tf.reshape(patches_fake[:,c,j,:], [-1,56,56,3])
            real_img_out = disc(real_img)
            fake_img_out = disc(fake_img)
            if fake:
                loss_real = loss_fn(tf.zeros_like(real_img_out), real_img_out)
                loss_fake = loss_fn(tf.ones_like(fake_img_out), fake_img_out)
            else: 
                loss_real = loss_fn(tf.ones_like(real_img_out), real_img_out)
                loss_fake = loss_fn(tf.zeros_like(fake_img_out), fake_img_out)
            losses.append((loss_real+loss_fake) / 2)
    complete_loss = tf.reduce_mean(losses)
    return complete_loss


@tf.function
def train_step(img):
    with tf.GradientTape() as disctape:
        gen_out = gen(img[0])
        complete_disc_loss = discriminator_loss([gen_out, img[1]]) / 2
    grads = disctape.gradient(complete_disc_loss, disc.trainable_weights)
    disc_opt.apply_gradients(zip(grads, disc.trainable_weights))

    with tf.GradientTape() as gentape:
        gen_out = gen(img[0])
        adv_loss_gen = discriminator_loss([gen_out, img[1]], fake=True)
        l1_loss = tf.reduce_sum(tf.math.abs(img[1] - gen_out))
        complete_gen_loss = 1e2*l1_loss + adv_loss_gen
    grads = gentape.gradient(complete_gen_loss, gen.trainable_weights)
    gen_opt.apply_gradients(zip(grads, gen.trainable_weights))
    return complete_gen_loss, complete_disc_loss

for _ in range(10):
    loss_array_gen = []
    loss_array_disc = []
    for steps, img in enumerate(dataset):
        complete_gen_loss, disc_loss = train_step(img)
        loss_array_gen.append(complete_gen_loss)
        loss_array_disc.append(disc_loss)
        if steps % 50 == 0:
            print("{0} in progress".format(_))
            print(f"steps: {steps}")       
            print("lossgen: " + str(complete_gen_loss))
            print("lossdisc: " + str(disc_loss))

    print("=======================")
    print("COMP_GEN_LOSS: {0}".format(np.mean(loss_array_gen)))
    print("COMP_DISC_LOSS: {0}".format(np.mean(loss_array_disc)))


    if _ % 1 == 0:
        img = gen(tf.expand_dims(test_img_,0),training=False) * 255
        img = tf.cast(tf.reshape(img,[224,224,3]), dtype=tf.uint8).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('{0}.jpg'.format(count), img)
        count += 1
gen.save("model/model1")