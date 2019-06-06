from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import visdom

from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
vis = visdom.Visdom(env='BeeNet')



def load_datas(data_dir='hymenoptera_data') :
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}


    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return dataloaders, dataset_sizes, device,class_names,image_datasets

def train_model(model,criterion,optimizer,scheduler,device,dataloaders,dataset_sizes,num_epochs=25) :

    # Save the first best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    loss_win = {'train':None,'val':None}
    for epoch in range(num_epochs) :
        print('Epochs {}/{}'.format(epoch,num_epochs-1))
        for phase in ['train','val'] :

            # Change the mode of the model
            if phase == 'train':
                scheduler.step()  # step the scheduler
                model.train()  # Set model to train mode

            if phase == 'val':
                model.eval()  # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase] :
                # Use gpu
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # Resets the grads

                with torch.set_grad_enabled(phase == 'train') :
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # get the predicted labels
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()  # set the grad in the net
                        optimizer.step()  # optimization step
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            #Visualization with visdom
            loss_win[phase] = vis.line([epoch_loss],[epoch],win=loss_win[phase],update='append' if loss_win[phase] else None,
                                        opts =dict(
                                        xlabel='Step',
                                        ylabel='Loss',
                                        title='{} Loss'.format(phase)))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc :
                # If we beat the best loss we save the model
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    # Load best model to return
    model.load_state_dict(best_model_wts)
    return model




if __name__ == '__main__' :

    dataloaders, dataset_sizes, device, _ ,_  = load_datas('hymenoptera_data')

    model_ft = models.resnet18(pretrained=True) # Load a resnet Model

    # We freeze all the layers of the original model
    #for params in model_ft.parameters():
    #   params.requires_grad = False


    # Reset the last Fully Connected Layer and change the output from 1000 to 2
    num_ftrs = model_ft.fc.in_features  # Get in features
    model_ft.fc = nn.Linear(num_ftrs, 2) # Replace te fc of the model with an empty linear with 2 layers

    # Load the model to GPU
    model_ft = model_ft.to(device)

    # Define the criterion optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.09)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    model_ft = train_model(model_ft, criterion, optimizer_ft,
                           exp_lr_scheduler,device,dataloaders,
                           dataset_sizes,num_epochs=25)





