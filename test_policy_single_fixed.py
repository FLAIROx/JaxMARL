import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="actor_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create fixed input vector of 0.1s
input_dim = input_details[0]['shape'][-1]
fixed_input = np.full((1, input_dim), 0.1, dtype=np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], fixed_input)
interpreter.invoke()
output_actions = interpreter.get_tensor(output_details[0]['index'])[0]

print("Output actions:", output_actions)
