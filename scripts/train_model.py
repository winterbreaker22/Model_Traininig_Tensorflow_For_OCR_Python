import tensorflow as tf
from object_detection import model_lib_v2
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import os
from object_detection.utils import config_util

def train_model(config_path, output_dir):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    config = config_util.create_configs_from_pipeline_proto(pipeline_config)

    train_input_fn = config['train_input_fn']
    eval_input_fn = config['eval_input_fn']

    model_config = config['model']
    checkpoint_dir = output_dir  

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_lib_v2.train_loop(
        pipeline_config=pipeline_config,
        model_dir=output_dir,  
        config=config,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn
    )

if __name__ == "__main__":
    config_path = f'{os.getcwd()}/model/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config' 

    output_dir = f'{os.getcwd()}/train_output'

    train_model(config_path, output_dir)

    print(f"Training completed. Outputs are saved to: {output_dir}")
