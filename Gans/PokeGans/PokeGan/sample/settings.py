import random as random
import torch
from visdom import Visdom

print('Load settings PokeGan')

# visualization dasboard

viz = Visdom(env='PokeGan')

# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

##### Data
#
# ​		`dataroot` : path of the dataset folder
dataroot = "../data"

# ​		`Image size ` + `nc` : Image size and number of channel [ nc, w , h ] - [3,64,64]
image_size = 128
nc = 3

# #### Computational Load
#
# ​		`workers`: Number of threads to load the data -2
workers = 2

# ​		`batch size` : Number of image / batch - 128
batch_size = 64

# ​		`ngpu` : available number of gpus - 1
ngpu = 1

# #### Model :
#
# ​		`nz` : Length of the latent vector  ( input of the generator) - 100
nz = 100

# ​		`ngf`: Depths of the feature map carried by the generator - 64

ngf = 32
# ​		`ndf` : Depths feature map discriminator - 64

ndf = 32

# #### Training :
#
# ​		`num_epoch` : Number of epoch to do - 5

num_epochs = 1000
# ​		`lr` : learning rate - 0.0002

lr = 0.00002

# ​		`beta1` : parameters for Adam optimizer - 0.5
beta1 = 0.5
