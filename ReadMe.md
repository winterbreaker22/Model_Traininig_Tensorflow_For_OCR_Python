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



# Clone the TensorFlow Models Repository

git clone https://github.com/tensorflow/models.git
cd models/research

pip install protobuf
python object_detection/packages/tf2/setup.py install



# Compile the Protobuf Files

protoc object_detection/protos/*.proto --python_out=.


