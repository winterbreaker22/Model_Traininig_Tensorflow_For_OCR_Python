import os
import sys
import tensorflow as tf
from object_detection import model_main_tf2
from absl import app

def train_model(pipeline_config_path, model_dir, num_train_steps=10000, use_tpu=False):
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    log_dir = os.path.join(model_dir, 'logs')
    eval_dir = os.path.join(model_dir, 'eval')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    if not os.path.exists(pipeline_config_path):
        raise FileNotFoundError(f"Pipeline config file not found: {pipeline_config_path}")

    sys.argv = [
        'model_main_tf2.py',  
        '--pipeline_config_path', pipeline_config_path,
        '--model_dir', model_dir,
        '--num_train_steps', str(num_train_steps),
        '--checkpoint_every_n', '1000', 
        '--eval_timeout', '300',  
        '--alsologtostderr', 
    ]

    if use_tpu:
        tpu_name = None  
        
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
    pipeline_config_path = f'{os.getcwd()}/pipeline.config' 
    model_dir = f'{os.getcwd()}/train_output' 
    num_train_steps = 20000  
    use_tpu = False 

    train_model(pipeline_config_path, model_dir, num_train_steps, use_tpu)
