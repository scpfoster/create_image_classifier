import argparse
import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

#### command line arguments for train.py
def train_cmd_line_def():
    parser = argparse.ArgumentParser(description='training a network and save a checkpoint',
                                     usage='python train.py \'data_directory\'')
    ## required argument of the data directory
    parser.add_argument('data_directory')
    ## optional parameters for the commandline
    parser.add_argument('--arch', action='store', dest='arch', default='densenet')
    parser.add_argument('--save_dir', action='store',
                        dest='save_dir', default='')
    parser.add_argument('--learning_rate', action='store',
                        dest='learn_rate', default=0.003)
    parser.add_argument('--hidden_units', nargs='*', action='store',
                        dest='hidden_units', type=int, default=[512, 256])
    parser.add_argument('--epochs', action='store', dest='epochs', default=8)
    parser.add_argument('--gpu', action='store_true',
                        dest='use_gpu', default=False)

    #error check on GPU available
    if parser.parse_args().use_gpu == True:
        check_4_gpu() 
        
    #error check on hidden units
    if len(parser.parse_args().hidden_units) != 2:
        print("--hidden_units only supports an array of size 2, size provided: ", len(parser.parse_args().hidden_units))
        exit()
        
    return parser.parse_args()

### command line arguments for predict.py
def predict_cmd_line_def():
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
    
    #error check on GPU available
    if parser.parse_args().use_gpu == True:
        check_4_gpu() 
        
    return parser.parse_args()
    
### check for gpu being available if specified
def check_4_gpu():
    if (torch.cuda.is_available()):
        print("gpu is available")
    else:
        print("gpu is not avaible, exiting")
        exit()     
             
    return torch.cuda.is_available()

### data_transform ###
## define the transforms to be used for training data and test,validation data
def create_data_transforms (training_transform):
    if (training_transform == 'train'):
        transform = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomResizedCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    else:
        transform = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    return transform

### data_load ###
## Open the data in the specified location and create the dataset and dataloader
def create_data_loader(file_dir, transforms, setShuffle=False, batch=64 ):
    
    image_datasets = datasets.ImageFolder(file_dir, transform=transforms)
    dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=batch, shuffle=setShuffle)

    return image_datasets, dataloader

### process image ###
## take in a path to an image for opening and processing.
## processing includes scaling, cropping and normalizing
## returns a tensor representing a 224 x 224 image
def process_image(cmd_inputs):
    
    image = Image.open(cmd_inputs.image_path)
    # resize the image so the shortest side is 256 pixels while maintaining aspect ratio
    if image.width < image.height:
        scaleFactor = image.width / 256
    else:
        scaleFactor = image.height / 256
    
    (width, height) = (int(image.width//scaleFactor), int(image.height//scaleFactor))
    
    tempImage = image.resize((width, height))
    
    # crop the centre 224 x 224 portion of the image
    # PIL coordinate system uses upper left as 0,0
    left = (width - 224)/2
    right = left + 224
    upper = (height - 224)/2
    lower = upper + 224
    
    tempImage = tempImage.crop((left, upper, right, lower))
    
    # convert to an np array, all the colour info is in the 3rd dimension
    tempImage = np.array(tempImage)
    
    # adjust the colour from 256 to floats between 0 and 1
    tempImage = tempImage/256
    
    # adjust for mean and std dev in the colour
    mean = [0.485, 0.456, 0.406]
    std_dev = [0.229, 0.224, 0.225]
    tempImage = (tempImage - mean)/std_dev
    
    #adjust the image dimensions
    tempImage = np.transpose(tempImage, (2, 0, 1))
    
    #convert to a pytorch tensor
    rtnImage = torch.from_numpy(tempImage)
    
    #return the processed image
    return rtnImage