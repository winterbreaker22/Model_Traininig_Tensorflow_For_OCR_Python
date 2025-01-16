import subprocess
import os
import sys

venv_path = os.path.join(os.getcwd(), 'venv', 'Scripts', 'activate_this.py')

# Path to the pipeline configuration
pipeline_config_path = os.path.join(os.getcwd(), 'model', 'pipeline.config')

# Path to the models directory (path to the "models" folder where cloned repo is located)
model_dir = os.path.join(os.getcwd(), 'model', 'training')

# Command to run the training process
train_command = [
    'python',
    os.path.join(os.getcwd(), 'models', 'research', 'object_detection', 'model_main_tf2.py'),
    '--pipeline_config_path', pipeline_config_path,
    '--model_dir', model_dir,
    '--num_train_steps', '5000',
    '--sample_1_of_n_eval_examples', '1',
    '--alsologtostderr'
]

# Activate virtual environment
def activate_venv():
    if not os.path.exists(venv_path):
        print(f"Virtual environment not found at: {venv_path}")
        sys.exit(1)
    
    # Activate the virtual environment
    exec(open(venv_path).read(), {'__file__': venv_path})

# Run the training command
def run_training():
    try:
        subprocess.run(train_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        sys.exit(1)
    else:
        print("Training started successfully!")

if __name__ == "__main__":
    # Activate virtual environment
    activate_venv()

    # Run training command
    run_training()
