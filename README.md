# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

The code in this repo reflects the code that was submitted for the project to build an image classifier using PyTorch.  The code was only ever run in the Udacity provided workspace using the Udacity provided GPU.  

The flowers directory in this repo does not contain meaningful data for training or assess the model.  It exists only to allow code debug to happen in my local environment.  The Udacity workspace provides a complete flowers directory with images for 102 classes.

## Jupyter Notebook

Files:
* Image Classifier Project.ipynb
* Image Classifier Project.html

The notebook uses the Torchvision Model VGG-16 with batch normalization.  The model was trained and tested over 8 epochs.  The accuracies associated with validation during training and post-training test were:
* validation during training - 76.8%
* testing post training - 76.0%

## Command line application

Files:
* train.py
* predict.py
* proj_model.py
* proj_utils.py

The command line application uses the Torchvision Model Densenet121 as the default model.  It also supports the VGG-16 with batch normalization model.  Two hidden layers are used for both the Densenet121 and VGG-16 models.

A checkpoint file is created after the model is trained and loaded.  The checkpoint file name will include a timestamp.

The model was only run on the Udacity provided GPU.  Only the default configuration was tested.

### usage
#### Train the Model
The file train.py is called with the top level data directory path and a number of optional inputs. The optional inputs are described in the proj_utils.py file.  The testing was completed with the following command line.
`python train.py flowers --gpu`

#### Predict a flower type
The file predict.py is called with the path to an image, the saved checkpoint file and a number of optional inputs.  The optional inputs are described in the proj_utils.py file.  The testing was completed with the following command line.
`python predict.py flowers/valid/1/image_06739.jpg checkpoint2020_06_07_18_44.pth --gpu --category_names=cat_to_name.json`

