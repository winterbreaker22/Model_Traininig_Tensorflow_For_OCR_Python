import os
import subprocess

# Set the paths for your model and config
pipeline_config_path = '/path/to/model/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config'
model_dir = '/path/to/model/my_model'

# TensorFlow Object Detection training command
command = [
    'python', 
    'models/research/object_detection/model_main_tf2.py', 
    '--pipeline_config_path={}'.format(pipeline_config_path),
    '--model_dir={}'.format(model_dir),
    '--alsologtostderr'
]

# Run the command
print("Starting training...")
subprocess.run(command, check=True)
print("Training complete!")
