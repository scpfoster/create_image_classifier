import torch
from torchvision import datasets, transforms, models
from torch import nn
from collections import OrderedDict
from torch import optim

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
        ('fc1', nn.Linear(input_size, int(cmd_input.hidden_units))),
        ('fc1_relu', nn.ReLU()),
        ('fc1_drop', nn.Dropout(0.3)),
        ('fc2', nn.Linear(int(cmd_input.hidden_units), 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    #replace the classifier
    model.classifier = new_classifier
    # return the model
    return model

def train_model(model, cmd_input, dataloaders):

    #define the optimizer, only for the classifier, learning rate of 0.003
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

    #define the criterion
    criterion = nn.NLLLoss()

    #define the device to use:
    if (cmd_input.use_gpu == True and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        print("cpu is being used either because gpu not specified OR cuda not available")
        print("get a coffee, this may take awhile")


    # counter of the number of steps through the model
    steps = 0
    # capture the correct and incorrect predications being made by the model
    running_loss = 0
    # check the loss and accuracy every 10 steps
    print_every = 10

    #make sure the model is in the correct place
    model.to(device)

    for epoch in range(cmd_input.epochs):
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
    
    return model
    
