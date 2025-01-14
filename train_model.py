import os
import tensorflow as tf
from object_detection import model_main_tf2

pipeline_config = '/model/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config'
model_dir = '/model/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint'

# Run training
model_main_tf2.main([
    '--pipeline_config_path', pipeline_config,
    '--model_dir', model_dir,
    '--alsologtostderr'
])
