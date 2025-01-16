

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
python object_detection/packages/tf2/setup.py install

Cmd
set PYTHONPATH=%cd%;%cd%\slim

Powershell
$env:PYTHONPATH = "$(Get-Location):$(Get-Location)\slim"


# Protoc Setup and compile

Download protoc.exe or zip file
Add protoc.exe to Env Variable

protoc object_detection/protos/*.proto --python_out=.


