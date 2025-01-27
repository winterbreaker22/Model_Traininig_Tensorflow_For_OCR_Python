import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("exported_model/saved_model")

# Enable Flex ops (to handle unsupported TensorFlow operations like StridedSlice)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Default TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS     # Enable TensorFlow ops (Flex delegate)
]

try:
    tflite_model = converter.convert()
    print("Model converted successfully!")
    
    with open("exported_model/model.tflite", "wb") as f:
        f.write(tflite_model)

except Exception as e:
    print(f"Error during conversion: {e}")

tf.lite.experimental.Analyzer.analyze(model_path='exported_model/model.tflite')