import random as random
import torch
from visdom import Visdom


# visualization dasboard
name_env = 'MicroGanV3'

print('Loading parameters {}'.format(name_env))

n_visualisation = 100

# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

##### Data
#
# ​		`dataroot` : path of the dataset folder
dataroot = "../data"
path_experiments = '../experiments/'

# ​		`Image size ` + `nc` : Image size and number of channel [ nc, w , h ] - [3,64,64]
image_size = 64
nc = 6

# #### Computational Load
#
# ​		`workers`: Number of threads to load the data -2
workers = 2

# ​		`batch size` : Number of image / batch - 128
batch_size = 256


# ​		`ngpu` : available number of gpus - 1
ngpu = 1

# #### Model :
#
# ​		`nz` : Length of the latent vector  ( input of the generator) - 100
nz = 100

# ​		`ngf`: Depths of the feature map carried by the generator - 64

ngf = 64
# ​		`ndf` : Depths feature map discriminator - 64

ndf = 64

# #### Training :
#
# ​		`num_epoch` : Number of epoch to do - 5

d_iters = 2
g_iters  = 1

# ​		`lr` : learning rate - 0.0002

lr = 5e-5

# ​		`beta1` : parameters for Adam optimizer - 0.5
beta1 = 0.5

# According to classical Wgan algorithm we clip the parameters of the descriminator
# so can netD() be a 1-Lipschitz function
c = 0.01