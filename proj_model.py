import torch
from torchvision import datasets, transforms, models
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import optim
from datetime import datetime
import numpy as np
import pandas as pd
import json

## will load the specified architecture and freeze the model parameters
## if architecture not supported, program will exit
def load_model_mod_classifier(cmd_input):
    if "vgg" in cmd_input.arch.lower():
        #code to load the vgg model
        model = models.vgg16_bn(pretrained=True)
        #get model inputs
        input_size = model.classifier[0].in_features

    elif "resnet" in cmd_input.arch.lower():
        # code to load a resnet model
        model = models.resnet18(pretrained=True)
        #get model inputs
        input_size = model.fc.in_features

    else:
        #unsupported model
        print("Only vgg and resnet are supported architectures at this time")
        exit()

    #freeze the model inputs
    for param in model.parameters():
        param.requires_grad = False

    #create new classifier
    new_classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, int(cmd_input.hidden_units[0]))),
        ('fc1_relu', nn.ReLU()),
        ('fc1_drop', nn.Dropout(0.3)),
        ('fc2', nn.Linear(int(cmd_input.hidden_units[0]), int(cmd_input.hidden_units[1]))),
        ('fc2_relu', nn.ReLU()),
        ('fc2_drop', nn.Dropout(0.3)),
        ('fc3', nn.Linear(int(cmd_input.hidden_units[1]), 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    #replace the classifier
    model.classifier = new_classifier
    # return the model
    return model

## will train the model and print out the progress every 10 steps while training
## only tested with the vgg16_bn model, but should support the resnet18 model as well
## only supports a single hidden layer with a commandline specified number of units
## only tested on the udacity GPU
## will train with the training dataset and test with the validation dataset
def train_model(model, cmd_input, dataloaders):

    #define the optimizer for the classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=cmd_input.learn_rate)

    #define the criterion
    criterion = nn.NLLLoss()

    #define the device to use:
    if (cmd_input.use_gpu == True and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        print("cpu is being used either because gpu not specified OR cuda not available")
        print("get a coffee, this may take a while")
        device = torch.device("cpu")


    # counter of the number of steps through the model
    steps = 0
    # capture the correct and incorrect predications being made by the model
    running_loss = 0
    # check the loss and accuracy every 10 steps
    print_every = 10

    #make sure the model is in the correct place
    model.to(device)

    for epoch in range(int(cmd_input.epochs)):
        for inputs, labels in dataloaders['train']:
            steps += 1
            # esnure labels and inputs are in the right location
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{cmd_input.epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                running_loss = 0
                model.train()
    
    return model, optimizer

## saves a checkpoint file of the specified model
## filename will include a timestamp in the name: checkpoint_yyyy_mm_mm_hh_mm
def save_checkpoint(model, cmd_input, dataset, optimizer):
    
    # create the checkpoint filename in the correct location:
    save_file = cmd_input.save_dir + 'checkpoint' + datetime.now().strftime("%Y_%m_%d_%H_%M") + '.pth'
    
    # set the model class_to_idx from the training dataset
    model.class_to_idx = dataset['train'].class_to_idx

    #set the arch
    if "vgg" in cmd_input.arch.lower():
        arch = 'vgg16_bn'
    elif "resnet" in cmd_input.arch.lower():
        arch = "resnet18"
    else:
        #unsupported model, this error should have been caught earlier
        #if for some reason it was not, just going to set to vgg16_bn
        #in real life, I would make this a stronger error check, but I 
        #have spent too much time on this already
        print("defaulting arch to vgg16_bn")
        arch = 'vgg16_bn'
    
    torch.save({
        'epochs': cmd_input.epoch,
        'arch': arch,
        'classifier': model.classifier,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }, save_file)
    
    return 

## loads the specified checkpoint and returns a model
## intended to be used to reload trained, or partially trained models
def load_checkpoint(checkpoint):
    #load checkpoint
    checkpoint = torch.load(file)
    #assign model passed on saved architecture
    if "resnet" in checkpoint['arch'].lower():
        model = models.resnet18(pretrained=True)
    else:
        #unsupported model, this error should have been caught earlier
        #if for some reason it was not, just going to set to vgg16_bn
        #in real life, I would make this a stronger error check, but I 
        #have spent too much time on this already
        model = models.vgg16_bn(pretrained=True)
        
    #update the model classifier, load weights and class_toidx
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

## from a pre-processed image tensor, makes a prediction on the image class using the specified model
## returns the specified number of most likely classes and their asscociated probabilities
## as a pandas series
def predict(model, img_tensor, cmd_input):
    
    #define the device to use:
    if (cmd_input.use_gpu == True and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        print("cpu is being used either because gpu not specified OR cuda not available")
        print("get a coffee, this may take awhile")
        
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        logps = model.forward(img_tensor)
        
    ps = torch.exp(logps)
    
    top_p, top_class = ps.topk(cmd_input.k_most_likely, dim=1)
    
    #adjust for 0 base vs 1 base counting
    top_class = top_class + 1
    
    #convert to numpy arrary
    top_p_np = np.array(top_p)[0]
    top_class_np = np.array(top_class)[0].astype('str')
    
    #if category mapping has been provided, map numbers to name
    
    #if a mapping for category names was provided
    if cmd_input.cat_names is not None:
        #open category to number mapping json
        with open(cmd_input.cat_names, 'r') as f:
            cat_to_name = json.load(f)

        #convert class names
        top_class_names = [cat_to_name[number] for number in top_class_np]
    
    return pd.Series(top_p_np, top_class_names)
    