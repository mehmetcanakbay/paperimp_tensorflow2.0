import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import cv2
import numpy as np
import matplotlib.pyplot as plt

percep_model = tf.keras.models.load_model("tflitemodel/perceptive")
mae_model = tf.keras.models.load_model("tflitemodel/psnr")
weights = percep_model.get_weights()
weights_mae = mae_model.get_weights()
new_weights = []
for i in range(len(weights)):
    weights[i] = weights[i] * 0.8
    weights_mae[i] = weights_mae[i] * 0.2
    tensor = tf.add(weights[i], weights_mae[i])
    new_weights.append(tensor)
    
percep_model.set_weights(new_weights)
model = percep_model
model.save("tflitemodel/interpolated")

