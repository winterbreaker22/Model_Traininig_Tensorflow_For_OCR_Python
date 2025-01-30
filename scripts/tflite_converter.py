import tensorflow as tf

saved_model_dir = "exported_model/saved_model"
original_model = tf.saved_model.load(saved_model_dir)

original_func = original_model.signatures["serving_default"]

@tf.function(input_signature=[
    tf.TensorSpec(shape=(1, 512, 512, 3), dtype=tf.uint8, name="input_tensor")
])
def new_serving_fn(input_tensor):
    return original_func(input_tensor)

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [new_serving_fn.get_concrete_function()]
)

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
