import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path='exported_model/model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print('Input Details:', input_details)
print('Output Details:', output_details)

# Test Inference
import numpy as np
input_data = np.zeros(input_details[0]['shape'], dtype=input_details[0]['dtype'])
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

for input_detail in input_details:
    input_data = interpreter.get_tensor(input_detail['index'])
    print('Input:', input_data.shape)

for output_detail in output_details:
    output_data = interpreter.get_tensor(output_detail['index'])
    print('Output:', output_data.shape)
