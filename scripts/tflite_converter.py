import tensorflow as tf

# Load the saved model
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model/my_model")

# Convert the model
tflite_model = converter.convert()

# Save the converted model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
