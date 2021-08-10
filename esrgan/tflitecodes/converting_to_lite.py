import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("model_sigmoid/interpolated")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open('model_dynamic.tflite', 'wb') as f:
    f.write(tflite_model)