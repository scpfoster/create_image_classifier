import argparse
import torch
from torchvision import datasets, transforms

#### command line arguments
def cmd_line_def():
    parser = argparse.ArgumentParser(description='training a network and save a checkpoint',
                                     usage='python train.py \'data_directory\'')
    ## required argument of the data directory
    parser.add_argument('data_directory')
    ## optional parameters for the commandline
    parser.add_argument('--arch', action='store', dest='arch', default='vgg')
    parser.add_argument('--save_dir', action='store',
                        dest='save_dir', default='./')
    parser.add_argument('--learning_rate', action='store',
                        dest='learn_rate', default=0.03)
    parser.add_argument('--hidden_units', action='store',
                        dest='hidden_units', default=512)
    parser.add_argument('--epochs', action='store', dest='epochs', default=10)
    parser.add_argument('--gpu', action='store_true',
                        dest='use_gpu', default=False)

    return parser.parse_args()

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
    dataset = datasets.ImageFolder(file_dir, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=setShuffle)

    return dataset, dataloader
