# Training Modules

They are the modules used to train Machine Learning models.\
They are mainly in Python because of its predominance in Machine Learning frameworks, but other languages can be used too.

Libraries to install:
pip install matplotlib
pip install torchvision
pip install datasets
pip install diffusers
pip install accelerate

in the PyMIDIMusic folder:
pip install -v ./ 

# Tips

Most ML libraries (Pytorch, TensorFlow) do not support GPU training on Windows, and will only train on CPU.
If you want the best performance and you are on windows, it is recommended to install linux on wsl for multi gpu training.
It can multiply training speed by 20 times.









