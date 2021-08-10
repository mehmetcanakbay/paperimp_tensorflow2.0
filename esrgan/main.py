import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from models import *
from utils import *
from dataprep import *
import time 
import matplotlib.pyplot as plt

datagen = get_iterable(batch_size=16)

inputs = tf.keras.layers.Input(shape=(None,None,3))
model = Generator()
# model = tf.keras.models.load_model('tflitemodel/psnr')
output = model(inputs)
generator = tf.keras.models.Model(inputs=inputs, outputs=output)
discriminator = Discriminator()
optimizer = tf.keras.optimizers.Adam(lr=1e-4)
optimizer_discriminator = tf.keras.optimizers.Adam(lr=5e-5)
feature_extractor = get_extractors()
count = 0
test_img_ = get_test_image()
testbatch, testwidth, testheight, testchannels = test_img_.shape
@tf.function
def train_step(low_res, high_res):
    with tf.GradientTape() as tape, tf.GradientTape() as tape2:
        gen_img = generator(low_res)
        percep_loss = perceptural_loss(high_res, gen_img, feature_extractor)
        #
        y_true_out = discriminator(high_res + (tf.random.normal(shape=tf.shape(high_res)) * 0.3))
        y_false_out = discriminator(gen_img + (tf.random.normal(shape=tf.shape(gen_img)) * 0.3))
        #
        relative_false = tf.nn.sigmoid(y_false_out - tf.reduce_mean(y_true_out))
        relative_true = tf.nn.sigmoid(y_true_out - tf.reduce_mean(y_false_out))
        adversarial_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(relative_false), relative_false))

        content_loss = tf.reduce_sum(tf.math.abs(high_res - gen_img))
        complete_gen_loss = percep_loss + (5e-3*adversarial_loss) + (1e-2*content_loss)

        discriminator_loss_true = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(relative_true), relative_true))
        discriminator_loss_false = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(relative_false), relative_false))
        complete_disc_loss = (discriminator_loss_false + discriminator_loss_true) / 2


    grads = tape.gradient(complete_gen_loss, generator.trainable_weights)
    optimizer.apply_gradients(zip(grads, generator.trainable_weights))

    grads_disc = tape2.gradient(complete_disc_loss, discriminator.trainable_weights)
    optimizer_discriminator.apply_gradients(zip(grads_disc, discriminator.trainable_weights))

    return complete_disc_loss, complete_gen_loss

for _ in range(100):
    for steps, (low_res, high_res) in enumerate(datagen):

        
        starttime = time.time()
        complete_disc_loss, complete_gen_loss= train_step(low_res, high_res)
        end_time = time.time()
        print("{0} in progress".format(_))
        print(f"steps: {steps}")       
        print("Disc loss: " + str(complete_disc_loss))
        print("Gen loss: " + str(complete_gen_loss))
        print("Elapsed Time: " + str(end_time- starttime))

        if steps>100:
            break
        
    if _ == 70:
        optimizer.lr.assign(5e-5)
    # if _ == 100:
    #     optimizer.lr.assign(2e-5)


    if _ % 2 == 0:
        plt.imshow(tf.cast(tf.reshape((generator(test_img_, training=False) *255), [212*2,212*2,3]),tf.uint8))
        plt.savefig("{0}.jpg".format(count))
        count += 1

# generator.save("models_gen/perceptive_model_sigmoid")
# generator.save("testmodels/gan")
generator.save("tflitemodel/perceptive")