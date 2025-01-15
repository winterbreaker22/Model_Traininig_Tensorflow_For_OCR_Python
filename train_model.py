import subprocess

def train_model():
    pipeline_config_path = "model/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config"
    model_dir = "model/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint"

    command = [
        "python", 
        "models/research/object_detection/model_main_tf2.py",
        "--pipeline_config_path", pipeline_config_path,
        "--model_dir", model_dir,
        "--num_train_steps", "10000"
    ]

    subprocess.run(command)

if __name__ == "__main__":
    train_model()
