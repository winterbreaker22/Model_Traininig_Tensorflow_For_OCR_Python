@ -1,38 +0,0 @@


# Python version 3.10

Create venv
venv/Scripts/activate

# Clone the TensorFlow Models Repository

git clone https://github.com/tensorflow/models.git
cd models/


# Packages install

pip install tensorflow-text==2.10.0
pip install tensorflow==2.15.0
pip install tensorflow-io==2.15.0

cd official
pip install -r requirements.txt

cd ..
cd research
set version of tf-models-official package as 2.15.0 in object_detection/packages/tf2/setup.py 
python object_detection/packages/tf2/setup.py install

Powershell
$env:PYTHONPATH="$env:PYTHONPATH;$PWD\models\research;$PWD\models\research\slim"
$env:TF_ENABLE_ONEDNN_OPTS="0"


# Protoc Setup and compile

Download protoc.exe or zip file
Add protoc.exe to Env Variable

protoc object_detection/protos/*.proto --python_out=.


# Scripts

For training

- Run preprocess.py to preprocess dataset
- Run split_dataset.py for splitting dataset for training and evaluating
- Run convert_xml_to_tfrecord.py to get tfrecord file
- Update pipeline.config to set train dataset dir, evaluate dataset dir, learning rate, image_resizer.....
- Make train_output folder
- Run train_model.py to train
- You will get checkpoints in train_output folder


# Export

- Run export_model.py


# Testing

- Run test_inference.py


# Model converting to tflite for mobile

- Run tflite_converter.py
- Run tflite_check.py to check input and output shape of tflite
