## Train a new network on a dataset and save the model as a checkpoint.
## Takes in a path to the directory holding the data.
## Prints out training loss, validation loss, and validation accuracy during training
## Will save a checkpoint of the trained model to a file called checkpoint.pth
##
## Only 2 models are supported: vgg16_bn and resnet18
## The code will take the specified model, define the last stage as a new classifier and
## train only the new classifier.  The new classier assumes the desired number of outputs
## will be 102.  It will also only implement 1 hidden layer.  The number of hidden units
## in the hidden layer will be 512 unless specified on the command line.
##
## The code assumes the data will have at 3 types of data: training, test and validation
##
## Required Command Line Arguments:
##  path to the data - as a string, eg: 'flowers'
## Optional Command Line Arguments:
##  --arch - the base model to be used, must be a torchvision supported model, 
##        default vgg
##  --save_dir - string - directory to save the file checkpoint.pth
##        default is current directory
##  --learning_rate - float - learning rate to be used during training
##        default is 0.03
##  --hidden_units - int - number of hidden units to use in the classifier
##        default is 512
##  --epochs - int - number of training epocs, default is 10
##  --gpu - use a GPU when available

import argparse
import proj_utils
import proj_model
from torchvision import datasets, transforms, models
from torch import optim

#import torch
#from torch import nn
#
#import torch.nn.functional as F

#from collections import OrderedDict

############
## set up commandline argument parsing
##########
cmd_input = proj_utils.cmd_line_def()

#########
## data paths
#########

train_dir = cmd_input.data_directory + '/train'
valid_dir = cmd_input.data_directory + '/valid'
test_dir = cmd_input.data_directory + '/test'
#########
## define transforms and create datasets
## use batch size of 64, shuffle the training data
########
train_tranforms = proj_utils.create_data_transforms('train')
test_valid_transforms = proj_utils.create_data_transforms('test_valid')

image_datasets = {}
dataloaders = {}
image_datasets['train'], dataloaders['train'] = proj_utils.create_data_loader(
    train_dir, train_tranforms, True)
image_datasets['test'], dataloaders['test'] = proj_utils.create_data_loader(
    test_dir, test_valid_transforms)
image_datasets['valid'], dataloaders['valid'] = proj_utils.create_data_loader(
    valid_dir, test_valid_transforms)

#####
## load the specified model and update the classifer passed on the hidden_units 
####
model = proj_model.load_model_mod_classifier(cmd_input)
#print(model)

#####
## train the model on the specified location with the specified hyper parameters
## note this has only ever been tested on the Udacity GPU
#######
trained_model = proj_model.train_model(model, cmd_input, image_dataset, image_dataloaders)

