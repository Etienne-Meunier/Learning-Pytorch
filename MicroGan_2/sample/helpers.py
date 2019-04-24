from settings import *
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch
from mapPrinter import showMapVisdom


def dense_to_one_hot(a) :
    """
    Return a one hot representation of the matrix
    :param a: torch matrix of shape (w,h)
    :return: ah : torch matrix of shape (nc,w,h)
    """
    _,w,h = a.size()
    ah = torch.zeros(nc,w,h)
    for c in range(nc) :
        ah[c, :, :] = (a == c)
    return ah


def one_hot_to_dense(a) :
    """
    Return a dense matrix by compacting a one hot matrix
    :param a: one hot matrix (nc,w,h)
    :return: dense matrix (w,h) with the labels
    """
    return a.argmax(0)

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample.unsqueeze(0).type(torch.FloatTensor)

def generate_index_matrix():
    idx = []
    for i in range(image_size) :
        for j in range(image_size) :
            idx.append([i,j])
    return torch.IntTensor(idx)

idxt = generate_index_matrix()


def display_city(mtx,win_name) :
    """
    Get a city and display a scatter plot representation on the visdom area
    """
    #print('Shape mtx : {} '.format(mtx.size()))
    mtd = one_hot_to_dense(mtx)
    viz.histogram(mtd.flatten(), win=win_name + ' Histogram mtd', opts=dict(title='Histogram mtd'))
    showMapVisdom(mtd, viz, generated=True, win_name=win_name)

def load_dataset() :
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.DatasetFolder(root=dataroot, transform=transforms.Compose([
                                                                        transforms.Lambda(dense_to_one_hot)
                                                                    ]),
                                 loader=npy_loader,
                                 extensions=['.npy'])
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Plot some training images
    real_batch = next(iter(dataloader))
    display_city(real_batch[0][0], win_name='Training Example')

    return dataset, dataloader ,device


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

