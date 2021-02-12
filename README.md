# TDCX - Assignment

#### Problem Statement - Create an object detection model and use it for prediction.

[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow 2.3](https://img.shields.io/badge/TensorFlow-2.3-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.3.0)

## Introduction

Tensorflow currently updated its [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) which is now compatible with Tensorflow 2,
I tried to train the a model published in the [TF2 model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md), and train them with custom data provided in mail.

This project have two parts
1. UI Part
2. Model training


## 1. Running the trained model

I have created a simple Streamlit powered app UI with Docker.
Why Streamlit? 
[Streamlit](https://docs.streamlit.io/en/stable/) is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science.

Why Docker? 
An end-to-end platform container means you have an IDE or Jupyter Notebook / Lab, and your entire working environment, running in the container and also run the code inside it.This provides you **Reproducibility** and **Portability**.

Steps to start the app:
```bash

git clone https://gitlab.com/maheshwariumang/tdc-assignment.git
cd tdc-assignment\streamlit_ui
docker build -t object_detection_app .
docker run -p 8501:8501 object_detection_app:latest

``` 
Then visit localhost:8501

Sample image for Streamlit App running at your 

![N|Solid](./images/streamlit_ui.jpg)

## 2. Using Tensorflow Object Detection API

Why Tensorflow Object detection API?
The [TensorFlow Object Detection API][tensorflowodapi] is an open source framework built on top of TensorFlow that makes it easy to construct, train and deploy object detection models.

How to use these pre-trained models?
Let's start installation of [TensorFlow Object Detection API][tensorflowodapi] and train custom detector model.
1. We will use [Anaconda](https://www.anaconda.com/products/individual) and create new [environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)for installing the packages.
```bash
# create new environment
conda create --name object_detection_env python=3.6

# activate your environment before installation or running your scripts 
conda activate object_detection_env
``` 
2. Installing the Tensorflow and its dependencies
```bash
# Installing tensorflow
conda install tensorflow==2.3.0 tensorflow-gpu==2.3.0

# installing cudatoolkit and cudnn
conda install cudatoolkit==10.1.243=h74a9793_0 
conda install cudnn==7.6.5=cuda10.1_0
``` 
3. Installing the Object Detection API framework
Clone the tensorflow models repository:
```bash
git clone https://github.com/tensorflow/models.git
```

Make sure you have [protobuf compiler](https://grpc.io/docs/protoc-installation/#install-using-a-package-manager) version >= 3.0, by typing `protoc --version`, or install it on Ubuntu by typing `apt install protobuf-compiler`.

Then proceed to the python package installation as follows:

```bash
# remember to activate your python environment first
cd models/research
# compile protos:
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API as a python package:
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```
This will install object detection api as python package that will be available in your environment.
4. Testing the installation
```bash
# Test the installation.
python object_detection/builders/model_builder_tf2_test.py
```

For more installation options, please refer to the original [installation guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md).

You will need some additional dependencies:

```bash
# install OpenCV python package
pip install opencv-python
pip install opencv-contrib-python
```
5. Preparing your custom dataset for training
Add train images to [train](./train) and test images to [test](./test)
```bash
python xml_to_csv.py
```
Creating a pbtxt file (This is required by object detection api)
```bash
python generate_pbtxt.py csv train_labels.csv label_map.pbtxt
```
Generating TFRecord files
```bash
python generate_tfreccords.py train_labels.csv annotations/label_map.pbtxt train train_tf_record.record
python generate_tfreccords.py test_labels.csv annotations/label_map.pbtxt test test_tf_record.record
```
6. Downloading the pre-trained model weights
I will be using SSD model with MobileNetV2 backbone as it is small model that can fit in a small GPU memory. You can check the other pretrained model with coco dataset in [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
```bash
# download the mobilenet_v2 model
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz
# extract the downloaded file
tar -xzvf ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz
```
I saved these these weights to [pre-trained-models](./pre-trained-models)
7. Configuring the pipeline configurations
Once the weights and config file is downloaded you can tweak pipeline.config parameters:

* Used `num_classes: 3` as we have only three class (apple, orange, banana), instead of 90 classes in coco dataset.
* Changed `fine_tune_checkpoint_type: "classification"` to `fine_tune_checkpoint_type: "detection"` as we are using the pre-trained detection model as initialization.
* Added the path of the pretrained model in the field `fine_tune_checkpoint:`, for example using the mobilenet v2 model I added `fine_tune_checkpoint: "./models/ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-0"`  
* Changed `batch_size: 512` and used a reasonable number to my GPU memory. I have a 4GB of GPU memory, so I am using `batch_size: 8`
* Added the maximum number of training iterations in `num_steps:`, and also used the same number in `total_steps:`
* Adapted the learning rate to our model and batch size (originally they used higher learning rates because they had bigger batch sizes). This values needs some testing and tuning.
* The `label_map_path:` should point to our labelmap file (here the raccoon labelmap) `label_map_path: "./annotations/labelmap.pbtxt"`
* You need to set the `tf_record_input_reader` under both `train_input_reader` and `eval_input_reader`. This should point to the tfrecords we generated (one for training and one for validation).
    ```
    train_input_reader: {
        label_map_path: "./annotations/labelmap.pbtxt"
        tf_record_input_reader {
            input_path: "../annotations/train.record"
        }
    }
    ``` 

8. Training the model
```bash
python model_main_tf2.py --model_dir=models/my_ssd_mobilenet_v2 \   
                         --pipeline_config_path=models/my_ssd_mobilenet_v2/pipeline.config \
                         --alsologtostderr
```
9. Check the training progress - you can use the Tensorboard to check the same

10. Exporting the trained model
```bash
python exporter_main_v2.py --input_type="image_tensor" \
                           --output_directory=streamlit_ui/exported_model \
                           --pipeline_config_path=models/my_ssd_mobilenet_v2/pipeline.config \
                           --trained_checkpoint_dir=models/my_ssd_mobilenet_v2/
```

## What can be done to improve the results?
- We can use the different pre-trained model as per the business need for higher accuracy
- We can use other Network like YoloV4 to detect the same
- We can fine tune the trained model by hyperparameter tuning.

# Credits:
- [Tensorflow][tensorflowodapi] team 
- TDCX for providing training and testing data

   [tensorflowodapi]: <https://github.com/tensorflow/models/tree/master/research/object_detection#tensorflow-object-detection-apir>
