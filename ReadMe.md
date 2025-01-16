packages to be installed

tensorflow
numpy
matplotlib
opencv-python
object-detection
labelImg



# Python version 3.10

pip install tensorflow
pip install tensorflow-io
pip install absl-py protobuf numpy pandas pillow lxml matplotlib opencv-python tensorflow-addons Cython contextlib2



# Clone the TensorFlow Models Repository

git clone https://github.com/tensorflow/models.git
cd models/research
python object_detection/packages/tf2/setup.py install

Powershell
$env:PYTHONPATH="$env:PYTHONPATH;$PWD\models\research;$PWD\models\research\slim"



# Compile the Protobuf Files

protoc object_detection/protos/*.proto --python_out=.


