from tensorflow.keras.applications.vgg19 import preprocess_input
import tensorflow as tf


def get_extractors():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    content = [
        'block1_conv1',
        'block2_conv2',
        'block3_conv3',
        'block4_conv3',
        'block5_conv4'
    ]
    outputs = [vgg.get_layer(name).output for name in content]
    feature_extractor_content = tf.keras.Model([vgg.input], outputs)
    # feature_extractor_content = tf.keras.Model(inputs = vgg.input, outputs= vgg.get_layer(content[0]).output)
    return feature_extractor_content

def perceptural_loss(y_true, y_hat, feature_extractor):
    batch, width, height, channels = y_true.shape 
    y_true = (y_true + 1) * 127.5
    y_hat = (y_hat + 1) * 127.5
    # y_true = y_true / 255
    # y_hat = y_hat / 255
    y_true = preprocess_input(y_true)
    y_hat = preprocess_input(y_hat)
    y_true_features = feature_extractor(y_true)
    y_hat_features = feature_extractor(y_hat)
    loss = 0
    for y_hat_feature, y_true_feature in zip(y_hat_features, y_true_features):
        y_true_feature = y_true_feature / 12.75
        y_hat_feature = y_hat_feature / 12.75
        loss += tf.reduce_sum((y_hat_feature - y_true_feature) ** 2) / (width * height)
    return loss

