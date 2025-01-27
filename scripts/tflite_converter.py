import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("exported_model/saved_model")

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,   
    tf.lite.OpsSet.SELECT_TF_OPS     
]

try:
    tflite_model = converter.convert()
    print("Model converted successfully!")
    
    with open("exported_model/model.tflite", "wb") as f:
        f.write(tflite_model)

except Exception as e:
    print(f"Error during conversion: {e}")

try:
    tf.lite.experimental.Analyzer.analyze(model_path='exported_model/model.tflite')
except Exception as e:
    print(f"Error during model analysis: {e}")
