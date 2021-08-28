from tensorflow.python.eager.backprop import GradientTape
from models import *
from utils import *
from dataprep import *
import tensorflow as tf
import time 
#I fixed the artifacts by using tanh activation at the end of the generator.

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

generator = Generator()
discriminator = Discriminator()
X, y = get_iterable()
steps = 30
optimizer = tf.keras.optimizers.Adam(lr=9e-4)
optimizer_discriminator = tf.keras.optimizers.Adam(lr=1e-4)
feature_extractor = get_extractors()
count = 0
# # high_res, low_res = get_only_one_img()
mse = tf.keras.losses.MeanSquaredError()

@tf.function
def train_step(low_res, high_res):
    with tf.GradientTape() as tape, tf.GradientTape() as tape2:
        gen_img = generator(low_res)
        content_loss = perceptural_loss(high_res, gen_img, feature_extractor)
        #
        y_true_out = discriminator(high_res)
        y_false_out = discriminator(gen_img)
        #
        adversarial_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(y_false_out), y_false_out)
        # adversarial_loss = -tf.math.log(y_false_out)
        mseloss = mse(high_res, gen_img)
        complete_gen_loss = content_loss + (1e-3*adversarial_loss)  + 1e-6*mseloss

        discriminator_loss_true = tf.keras.losses.binary_crossentropy(tf.ones_like(y_true_out), y_true_out)
        discriminator_loss_false = tf.keras.losses.binary_crossentropy(tf.zeros_like(y_false_out), y_false_out)
        complete_disc_loss = (discriminator_loss_false + discriminator_loss_true) / 2

    grads = tape.gradient(complete_gen_loss, generator.trainable_weights)
    optimizer.apply_gradients(zip(grads, generator.trainable_weights))

    grads_disc = tape2.gradient(complete_disc_loss, discriminator.trainable_weights)
    optimizer_discriminator.apply_gradients(zip(grads_disc, discriminator.trainable_weights))

    return complete_disc_loss, complete_gen_loss

for _ in range(30):
    for low_res, high_res in zip(X, y):

        print("{0} in progress".format(_))       
        starttime = time.time()
        complete_disc_loss, complete_gen_loss = train_step(low_res, high_res)
        end_time = time.time()
        print("Disc loss: " + str(complete_disc_loss))
        print("Gen loss: " + str(complete_gen_loss))
        print("Elapsed Time: " + str(end_time- starttime))
    
    if _ == 50:
        optimizer.lr.assign(1e-5)

    if _ % 5 == 0:
        plt.imshow(tf.cast(tf.reshape((generator(tf.expand_dims(low_res[0], axis=0), training=False) + 1) * 127.5, [384,384,3]),tf.uint8))
        plt.savefig("{0}.jpg".format(count))
        count += 1

generator.save_weights("saved_weights4/we1")


