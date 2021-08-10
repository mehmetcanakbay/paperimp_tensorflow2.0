import tflite_runtime.interpreter as tflite
import cv2
import numpy as np

interpreter = tflite.Interpreter(model_path="model_fp16.tflite", num_threads=6)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#1920x1080 image
img = cv2.imread(r"mordor.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
w,h,c = img.shape
img = cv2.resize(img, (h//2, w//2))
img = img/255
img = img.astype('float32')
img = np.expand_dims(img, axis=0)
interpreter.resize_tensor_input(
    input_details[0]['index'], (1, w//2, h//2, c))

print("hey1")
interpreter.allocate_tensors()
print("hey2")
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
print("hey3")
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data.shape)
output_data = cv2.cvtColor(output_data[0], cv2.COLOR_RGB2BGR)
print(output_data.shape)
output_data = (output_data*255).astype('uint8')
print(output_data.shape)

cv2.imwrite("tfliteoutput.jpg", output_data)

