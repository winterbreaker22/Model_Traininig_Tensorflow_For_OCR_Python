import tensorflow as tf
from object_detection import model_lib_v2
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import os
from object_detection.utils import config_util

def train_model(config_path, output_dir):
    # Load pipeline config
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    # Get the model config from pipeline config
    config = config_util.create_configs_from_pipeline_proto(pipeline_config)

    # Set up the train and eval input function
    train_input_fn = config['train_input_fn']
    eval_input_fn = config['eval_input_fn']

    # Set up the model and checkpoint directory
    model_config = config['model']
    checkpoint_dir = output_dir  # Use output_dir to store checkpoints

    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Start training using the model_lib_v2
    model_lib_v2.train_loop(
        pipeline_config=pipeline_config,
        model_dir=output_dir,  # Store training output in the specified output directory
        config=config,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn
    )

if __name__ == "__main__":
    # Path to your pipeline config file
    config_path = '/path/to/pipeline.config'  # Change this path to your pipeline.config file

    # Specify the output directory for training results (train_output in the root directory)
    output_dir = 'train_output'  # Change this path to your desired output directory (e.g., /train_output)

    # Start training and save the results to the output directory
    train_model(config_path, output_dir)

    print(f"Training completed. Outputs are saved to: {output_dir}")
