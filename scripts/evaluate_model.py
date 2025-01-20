import os
import sys
import tensorflow as tf
from object_detection import model_main_tf2
from absl import app

def evaluate_model(pipeline_config_path, model_dir, checkpoint_dir, use_tpu=False):
    os.makedirs(model_dir, exist_ok=True)
    eval_dir = os.path.join(model_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)

    if not os.path.exists(pipeline_config_path):
        raise FileNotFoundError(f"Pipeline config file not found: {pipeline_config_path}")

    # Modify sys.argv to call model_main_tf2.py with evaluation flags
    sys.argv = [
        'model_main_tf2.py', 
        '--pipeline_config_path', pipeline_config_path,
        '--model_dir', model_dir,
        '--checkpoint_dir', checkpoint_dir,  # Specify the correct checkpoint directory
        '--eval_timeout', '300',
        '--alsologtostderr',  # This will log to stderr as well
    ]

    if use_tpu:
        tpu_name = None  # Define your TPU name here, if using one
        
        if tpu_name is None:
            raise ValueError("Please provide a TPU Name to connect to.")
        
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_name)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
    else:
        strategy = tf.compat.v2.distribute.MirroredStrategy()

    app.run(model_main_tf2.main)

if __name__ == '__main__':
    pipeline_config_path = f'{os.getcwd()}/model/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config'
    model_dir = f'{os.getcwd()}/train_output'
    checkpoint_dir = f'{os.getcwd()}/train_output'  # This should be the directory with model checkpoints
    use_tpu = False  # Set True if using TPU

    evaluate_model(pipeline_config_path, model_dir, checkpoint_dir, use_tpu)
