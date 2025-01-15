packages to be installed

tensorflow
numpy
matplotlib
opencv-python
object-detection
labelImg




# Clone the TensorFlow Models Repository

git clone https://github.com/tensorflow/models.git
cd models/research



# Compile the Protobuf Files

protoc object_detection/protos/*.proto --python_out=.



# tensorflow-io package install

https://pypi.org/simple/tensorflow-io/
Download .whl file from this url

pip install tensorflow_io-*.whl
