import os
import subprocess

# Set paths for evaluation
pipeline_config_path = '/path/to/model/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config'
model_dir = '/path/to/model/my_model'
checkpoint_dir = '/path/to/model/my_model/checkpoint'

# TensorFlow Object Detection evaluation command
command = [
    'python', 
    'models/research/object_detection/model_main_tf2.py', 
    '--pipeline_config_path={}'.format(pipeline_config_path),
    '--model_dir={}'.format(model_dir),
    '--checkpoint_dir={}'.format(checkpoint_dir),
    '--alsologtostderr',
    '--eval'
]

# Run the evaluation
print("Starting evaluation...")
subprocess.run(command, check=True)
print("Evaluation complete!")
