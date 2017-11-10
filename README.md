# 204 Numerical Methods Project
This is an application built for the MTE 204 Numerical Methods course project. For this project each group was to apply various numerical methods to solve a practical engineering problem. The problem our group chose to solve was simulating mechanical depth of field using only software. The current version of this project was developed by:
Emma Xie, Emily Cho, and Kaitlyn Mccluskie

## Installation
First, clone project
```
git clone https://github.com/eacho1/img-processing-project.git
```
### Setting up Virtual environment
1. Install virtualenv `$ [sudo] pip install virtualenv`
2. Create new virtualenv `$ virtualenv ~/mte204-group-9`
..* a virtual environment is created in the home directory with name mte204-group-9
3. activate script `source ~/mte204-group-9/bin/activate`
4. To exit virtual environment later, simply `deactivate`

### Install Tensorflow, Keras, image segmentation library 
(Modified from the installation guide of [CRF-RNN for Semantic Image Segmentation](https://github.com/sadeepj/crfasrnn_keras/blob/master/README.md)) 

Install Image Segmentation library
```
git clone --recursive https://github.com/torrvision/crfasrnn.git
```

Install [Tensorflow](https://www.tensorflow.org/install/) and [Keras](https://keras.io/#installation), following the respective installation guides. You will need to install Keras with HDF5/h5py if you plan to use the provided trained model. 
After installing, run the following commands to make sure they are properly installed:
```
$ source ~/mte204-group-9/bin/activate  
$ python
>>> import tensorflow
>>> import keras
```
You should not see any errors while importing `tensorflow` and `keras` above.

### Build CRF-RNN custom C++ code

Checkout the code in this repository, activate the Tensorflow/Keras virtualenv (if you used one), and run the `compile.sh` script in the `cpp` directory. That is, run the following commands:
```
$ cd ~/mte204-group-9
$ git clone https://github.com/sadeepj/crfasrnn_keras.git
$ cd crfasrnn_keras/cpp
$ source ~/mte204-group-9/bin/activate
$ ./compile.sh
``` 
If the build succeeds, you will see a new file named `high_dim_filter.so`. If it fails, please see the comments inside the `compile.sh` file for help. You could also refer to the official Tensorflow guide for [building a custom op](https://www.tensorflow.org/extend/adding_an_op#build_the_op_library).

*Note*: This script will not work on Windows OS. If you are on Windows, please check [this issue](https://github.com/tensorflow/models/issues/1103) and the comments therein. The official Tensorflow guide for building a custom op does not yet include build instructions for Windows.

### Download the pre-trained model weights

Download the model weights from [here](https://goo.gl/ciEYZi) and place it in the `crfasrnn_keras` directory with the file name `crfrnn_keras_model.h5`.

### Install other Python dependencies
```
pip install -r requirements.txt
```
### Run the demo
```
$ cd crfasrnn_keras
$ python run_demo.py  # Make sure that the correct virtualenv is already activated
```

Wait about a minute for the first image to appear. The application will apply the blurring to all photos in `crfasrnn_keras/images`

