@ -1,38 +0,0 @@


# Python version 3.10

# Clone the TensorFlow Models Repository

git clone https://github.com/tensorflow/models.git
cd models/


# Packages install

pip install tensorflow-text==2.10.0
pip install tensorflow==2.10.1
pip install tensorflow-io

cd official
pip install -r requirements.txt

cd ..
cd research
in object_detection/packages/tf2/setup.py set version of tf-models-official package
python object_detection/packages/tf2/setup.py install

Powershell
$env:PYTHONPATH="$env:PYTHONPATH;$PWD\models\research;$PWD\models\research\slim"


# Protoc Setup and compile

Download protoc.exe or zip file
Add protoc.exe to Env Variable

protoc object_detection/protos/*.proto --python_out=.

