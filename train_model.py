import os

# Define paths
PIPELINE_CONFIG_PATH = "/model/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config"
MODEL_DIR = "/model/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
NUM_TRAIN_STEPS = 10000

# Construct training command
train_command = (f"python models/research/object_detection/model_main_tf2.py "
                 f"--pipeline_config_path={PIPELINE_CONFIG_PATH} "
                 f"--model_dir={MODEL_DIR} "
                 f"--num_train_steps={NUM_TRAIN_STEPS} "
                 f"--alsologtostderr")

# Run training command
os.system(train_command)
