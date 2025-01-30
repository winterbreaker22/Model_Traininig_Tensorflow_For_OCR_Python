import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



model = tf.saved_model.load("exported_model/saved_model")

signature_keys = list(model.signatures.keys())
print("Available Signatures:", signature_keys)

infer = model.signatures["serving_default"]

print("\nInputs:")
for input_name, input_tensor in infer.structured_input_signature[1].items():
    print(f"  Input Name: {input_name}, Shape: {input_tensor.shape}, Dtype: {input_tensor.dtype}")

print("\nOutputs:")
for output_name, output_tensor in infer.structured_outputs.items():
    print(f"  Output Name: {output_name}, Shape: {output_tensor.shape}, Dtype: {output_tensor.dtype}")
