import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from models import *
from dataprep import *
import time 


inputs = tf.keras.layers.Input(shape=(None, None, 3))
model = Generator()
output = model(inputs)
generator = tf.keras.models.Model(inputs=inputs, outputs=output)
optimizer_generator = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.5, beta_2=0.999)
discriminator = Discriminator()
optimizer_discriminator = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.5, beta_2=0.999)
count = 0
test_img_ = get_test_image()
testbatch, testwidth, testheight, testchannels = test_img_.shape
datagen = get_iterable()

@tf.function
def train_step(x_input, y_input):
    #Train discriminator
    with tf.GradientTape() as tape:
        gen_img = generator(x_input)
        y_true_out = discriminator(y_input + (tf.random.normal(shape=tf.shape(y_input)) * 0.2)) #-> fuzzy inputs (noise for disc)
        y_false_out = discriminator(gen_img + (tf.random.normal(shape=tf.shape(gen_img)) * 0.2))
        y_t_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(y_true_out), y_true_out, from_logits=True))
        y_f_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(y_false_out), y_false_out, from_logits=True))
        disc_loss = (y_f_loss + y_t_loss) / 2
    grads = tape.gradient(disc_loss, discriminator.trainable_weights)
    optimizer_discriminator.apply_gradients(zip(grads, discriminator.trainable_weights))
    #Train generator
    with tf.GradientTape() as tape2:
        gen_img = generator(x_input)
        y_false_out = discriminator(gen_img + (tf.random.normal(shape=tf.shape(gen_img)) * 0.2))
        adv_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(y_false_out), y_false_out, from_logits=True))
        l1_loss = tf.reduce_sum(tf.math.abs(y_input - gen_img)) / 4 # 4-> batch size
        complete_gen_loss = 100*l1_loss + adv_loss
    
    grads = tape2.gradient(complete_gen_loss, generator.trainable_weights)
    optimizer_generator.apply_gradients(zip(grads, generator.trainable_weights))

    return complete_gen_loss, disc_loss


for _ in range(300):
    loss_array_gen = []
    loss_array_disc = []
    for steps, (x_input, y_input) in enumerate(datagen):
        complete_gen_loss, disc_loss = train_step(x_input, y_input)
        loss_array_gen.append(complete_gen_loss)
        loss_array_disc.append(disc_loss)
        if steps % 50 == 0:
            print("{0} in progress".format(_))
            print(f"steps: {steps}")       
            print("lossgen: " + str(complete_gen_loss))
            print("lossdisc: " + str(disc_loss))

        if steps>100:
            break

    print("=======================")
    print("COMP_GEN_LOSS: {0}".format(np.mean(loss_array_gen)))
    print("COMP_DISC_LOSS: {0}".format(np.mean(loss_array_disc)))


    if _ % 2 == 0:
        img = ((generator(test_img_, training=False) - 1) *127.5) + 127.5
        img = tf.cast(tf.reshape(img,[256,256,3]), dtype=tf.uint8).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('{0}.jpg'.format(count), img)
        count += 1

"""
x_image = cv2.cvtColor((x_input[0] * 255).astype("uint8").reshape(192,192,3), cv2.COLOR_RGB2BGR)
cv2.imwrite('test_X.jpg'.format(count), x_image)
x_image = cv2.cvtColor((x_canny[0] * 255).astype("uint8").reshape(192,192,1), cv2.COLOR_RGB2BGR)
cv2.imwrite('test_X_CANNY.jpg'.format(count), x_image)
x_image = cv2.cvtColor((y_input[0] * 255).astype("uint8").reshape(192,192,3), cv2.COLOR_RGB2BGR)
cv2.imwrite('test_Y.jpg'.format(count), x_image)
"""