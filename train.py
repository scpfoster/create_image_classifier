import argparse
import torch

## set up commandline argument parsing
parser = argparse.ArgumentParser(description='training a network and save a checkpoint', 
                                 usage='python train.py \'data_directory\'')

## required argument of the data directory
parser.add_argument('data_directory')
## optional parameters for the commandline
parser.add_argument('--arch', action='store', dest='arch', default='vgg16_bn')
parser.add_argument('--save_dir', action='store', dest='save_dir', default = './')
parser.add_argument('--learning_rate', action='store', dest='learn_rate', default = 0.03)
parser.add_argument('--hidden_units', action='store', dest='hidden_units')
parser.add_argument('--epocs', action='store', dest='epocs', default = 10)
parser.add_argument('--gpu', action='store_true', dest='use_gpu', default=False)

cmd_input = parser.parse_args()

print(cmd_input)