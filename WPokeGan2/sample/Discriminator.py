from settings import *
import torch.nn as nn
from WPokeGan2.sample.helpers import same_padding_conv2

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(in_channels=nc,out_channels=ndf,kernel_size=5,stride=2,padding=same_padding_conv2(nc,5,2)),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(True),
            #
            nn.Conv2d(ndf, 2*ndf, 5, 2, same_padding_conv2(nc, 5, 2)),
            nn.BatchNorm2d(2*ndf),
            nn.LeakyReLU(True),
            #
            nn.Conv2d(2*ndf, 4*ndf, 5, 2, same_padding_conv2(nc, 5, 2)),
            nn.BatchNorm2d(4*ndf),
            nn.LeakyReLU(True),
            #
            nn.Conv2d(4*ndf, 8*ndf, 5, 2, same_padding_conv2(nc, 5, 2)),
            nn.BatchNorm2d(8*ndf),
            nn.LeakyReLU(True),

            ## Flatten
            ## Dense Layer with linear activation

        )

    def forward(self, input):
        return self.main(input)
