import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("CUDA and cuDNN are installed and working.")
else:
    print("CUDA/cuDNN not detected.")
