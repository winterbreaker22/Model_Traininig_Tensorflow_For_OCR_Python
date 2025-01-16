

# Python version 3.10

# Packages install

pip install tensorflow
pip install tensorflow-io
pip install numpy pandas pillow lxml matplotlib opencv-python tensorflow-addons Cython contextlib2 gin-config pyyaml



# Clone the TensorFlow Models Repository

git clone https://github.com/tensorflow/models.git
cd models/research
python object_detection/packages/tf2/setup.py install

Powershell
$env:PYTHONPATH="$env:PYTHONPATH;$PWD\models\research;$PWD\models\research\slim"



# Protoc Setup and compile

Download protoc.exe or zip file
Add protoc.exe to Env Variable

protoc object_detection/protos/*.proto --python_out=.


