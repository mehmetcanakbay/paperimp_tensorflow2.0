from numpy.core.shape_base import block
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input

def get_extractors():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    block_name = 'block5_conv4'
    copy_layer = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
    model = tf.keras.models.Sequential(vgg.layers[:20] + [copy_layer])
    copy_layer.set_weights(vgg.layers[20].get_weights())
    model.trainable = False
    return model
    

def perceptural_loss(y_true, y_hat, feature_extractor):
    y_true = (y_true *255) 
    y_hat = (y_hat *255) 
    y_true = preprocess_input(y_true)
    y_hat = preprocess_input(y_hat)
    y_true_features = feature_extractor(y_true) 
    y_hat_features = feature_extractor(y_hat)
    batch, width, height, channels = y_hat_features.shape 
    y_true_feature = y_true_features / 12.75
    y_hat_feature = y_hat_features / 12.75
    loss = tf.reduce_sum(tf.math.abs(y_true_feature - y_hat_feature)) / (width*height)
    return loss
