## Predict flower name from an image along with the probability of that name.
## Takes in a single image path and returns the flower name and class probability.
## Required Command Line Arguments:
##  path to the image - as a string, eg: '/data/2/img1.jpg'
##  path to a saved checkpoint from a trained model - as a string, eg: 'checkpoint.pth'
## Optional Command Line Arguments:
##  --top_k - the number of k most likely classes to return
##  --category_names - dictionary to map category numbers to real names eg cat_to_name.json
##  --gpu - use a GPU when available

import proj_utils
import proj_model
import pandas as pd
import torch
import json

# bring in the command line options
cmd_input = proj_utils.predict_cmd_line_def()
#print(cmd_input)


# load a model from a saved checkpoint
print("Load the model from the checkpoint file")
model = proj_model.load_checkpoint(cmd_input.checkpoint_file)

# process the specified image 
print("process the image")
img_tensor = proj_utils.process_image(cmd_input)

#predict the k most likely classes and probabilities
print("predict the image class and associated probability")
predicted_class = proj_model.predict(model, img_tensor, cmd_input)

# print the actual class if mapping provided
actual_class = cmd_input.image_path.split('/')[2]
if cmd_input.cat_names is not None:
    #open category to number mapping json
    with open(cmd_input.cat_names, 'r') as f:
        cat_to_name = json.load(f)
    actual_class = cat_to_name[actual_class]       
print("The acutal class of the image is:", actual_class)
          
# print the predicted classes
print("The ", cmd_input.k_most_likely, " classes and associated proabilities are below")
print(predicted_class)
