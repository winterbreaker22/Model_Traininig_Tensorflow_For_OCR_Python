import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import dataset_util
from object_detection.utils import model_util
from object_detection.utils import visualization_utils as vis_util
import os

def create_input_fn(tfrecord_path, label_map_path, batch_size=4):
    """Creates an input function for loading the dataset."""
    def input_fn():
        # Read and parse TFRecord file
        dataset = dataset_util.make_dataset_from_tfrecords(tfrecord_path, label_map_path)
        dataset = dataset.batch(batch_size)
        return dataset
    return input_fn

def build_model(pipeline_config_path):
    """Builds the model from the pipeline config."""
    configs = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(pipeline_config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, configs)
    
    model_config = configs.model
    model = model_util.create_model(model_config)  # Create the model using the pipeline config
    return model, configs

def train_model(pipeline_config_path, train_record_path, eval_record_path, label_map_path, model_dir):
    # Load the model and configuration
    model, configs = build_model(pipeline_config_path)

    # Create input functions for train and eval datasets
    train_input_fn = create_input_fn(train_record_path, label_map_path)
    eval_input_fn = create_input_fn(eval_record_path, label_map_path)

    # Optimizer and loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=configs.train_config.optimizer.adam_learning_rate)

    # Checkpoint manager
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint=tf.train.Checkpoint(model=model, optimizer=optimizer),
        directory=model_dir,
        max_to_keep=5
    )

    # Training loop
    for epoch in range(configs.train_config.num_steps):
        for step, (images, labels) in enumerate(train_input_fn()):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = compute_loss(predictions, labels)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.numpy()}")

        checkpoint_manager.save()  # Save the checkpoint after each epoch

        # Run evaluation after each epoch
        eval_results = evaluate_model(model, eval_input_fn)
        print(f"Epoch {epoch}, Evaluation Results: {eval_results}")

def compute_loss(predictions, labels):
    """Compute loss (example with mean squared error)."""
    loss = tf.reduce_mean(tf.square(predictions - labels))
    return loss

def evaluate_model(model, eval_input_fn):
    """Evaluate the model."""
    eval_loss = 0.0
    num_steps = 0

    for images, labels in eval_input_fn():
        predictions = model(images, training=False)
        eval_loss += compute_loss(predictions, labels)
        num_steps += 1

    return eval_loss / num_steps  

if __name__ == '__main__':
    # File paths
    pipeline_config_path = 'model/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config'
    train_record_path = 'dataset/train.record'
    eval_record_path = 'path/to/eval.record'
    label_map_path = 'path/to/label_map.pbtxt'
    model_dir = 'path/to/model_dir'

    os.makedirs(model_dir, exist_ok=True)

    # Train the model
    train_model(pipeline_config_path, train_record_path, eval_record_path, label_map_path, model_dir)
