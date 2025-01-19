import os
import tensorflow as tf
from object_detection.model_lib import eval_loop
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

def main():
    pipeline_config_path = f"{os.getcwd()}/model/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config"
    model_dir = f"{os.getcwd()}/train_output"  

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(pipeline_config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    configs = config_util.create_configs_from_pipeline_proto(pipeline_config_path)

    eval_loop(pipeline_config, model_dir)

if __name__ == "__main__":
    main()
