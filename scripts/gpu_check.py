import tensorflow as tf

print ("tensorflow version: ", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("CUDA and cuDNN are installed and working.")
else:
    print("CUDA/cuDNN not detected.")

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# Create a tensor and perform a simple computation
# with tf.device('/GPU:0'):  # Explicitly specify GPU device
#     a = tf.random.normal([1000, 1000])
#     b = tf.random.normal([1000, 1000])
#     c = tf.matmul(a, b)

# print(c)
