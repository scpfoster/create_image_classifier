## Predict flower name from an image along with the probability of that name.
## Takes in a single image path and returns the flower name and class probability.
## Required Command Line Arguments:
##  path to the image - as a string, eg: '/data/Lily/img1.jpg'
##  path to a saved checkpoint from a trained model - as a string, eg: './checkpoint.pth'
## Optional Command Line Arguments:
##  --top_k - the number of k most likely classes to return
##  --category_names - dictionary to map category numbers to real names eg cat_to_name.json
##  --gpu - use a GPU when available

import argparse

## set up commandline argument parsing
parser = argparse.ArgumentParser(description='predict type of flower in an image',
                                 usage='python predict.py \'data\/Lily\/img1.jpg\' \'checkpint.pth\'')

## required argument of the data directory
parser.add_argument('image_path')
parser.add_argument('checkpoint_file')
## optional parameters for the commandline
parser.add_argument('--top_k', action='store', dest='k_most_likely', default='5')
parser.add_argument('--category_names', action='store', dest='cat_names')
parser.add_argument('--gpu', action='store_true', dest='use_gpu', default=False)

cmd_input = parser.parse_args()

print(cmd_input)
