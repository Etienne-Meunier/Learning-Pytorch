from settings import *
import torch.nn as nn
from helpers import load_dataset,weights_init
from Discriminator import Discriminator
from Generator import Generator
import torch.optim as optim

class WPokeGan() :

    def __init__(self):
        self.dataset, self.dataloader, self.device = load_dataset()

        self.netG = Generator(ngpu).to(self.device)
        self.netD = Discriminator(ngpu).to(self.device)

        # Initialise Weights
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)


        # define loss function
        self.criterion = nn.BCELoss()

        # We create a fixed subset of random for the latent variable, this way we can evauate our progress.
        self.fixed_noise = torch.randn(64, nz, 1, 1, device=self.device)

        # Establish convention for real and fake labels during training
        self.real_label = 1
        self.fake_label = 0

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.RMSprop(self.netD.parameters(), lr=lr)
        self.optimizerG = optim.RMSprop(self.netG.parameters(), lr=lr)


        # Fixed noise for visualisation
        self.fixed_noise = torch.randn(64, nz, 1, 1, device=self.device)

    def train(self):
        # Training Loop


        iters = 0

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):

                # Reset the grad for the Discriminator Network
                self.netD.zero_grad()

                # Format batch
                real_cpu = data[0].to(self.device)  # Load the batch to gpu

                # Create the vector of the input size with only True = [real_label]*b_size
                b_size = real_cpu.size(0)

                # Training the Discriminator
                for i_d in range(d_iters) :
                    # Generate Results for real images
                    real_results = self.netD(real_cpu).view(-1)
                    D_x = real_results.mean().item()  # Accuracy of the classifier for True examples

                    # Generate Results for fake images
                    noise = torch.randn(b_size, nz, 1, 1, device=self.device)
                    fake_images = self.netG(noise)
                    fake_results = self.netD(fake_images.detach()).view(-1)

                    D_G_z1 = fake_results.mean().item()  # G sucess faking

                    # Compute Discriminator Loss using Wasserstein Formula - la elle est un peu chelou a verifier
                    errD = fake_results.mean() - real_results.mean()

                    self.netD.zero_grad()
                    errD.backward()
                    self.optimizerD.step()

                    # Clipping the weights of the discriminator
                    for p in self.netD.parameters() :
                        p.data.clamp_(-c ,c)

                # Training the Generator
                for i_g in range(g_iters) :
                    # Discriminator prediction
                    fake_results = self.netD(fake_images).view(-1)
                    errG = - fake_results.mean()
                    D_G_z2 = fake_results.mean().item()  # G sucess faking

                    self.netG.zero_grad()
                    errG.backward()
                    self.optimizerG.step()

                iters +=1
                self.visualize_progession(i, iters, epoch, errD, errG, D_x, D_G_z1, D_G_z2)


    def visualize_progession(self,i,iters,epoch,errD,errG,D_x,D_G_z1,D_G_z2) :
        """
        Handle the printing and plotting of the loss
        :return:
        """
        # Output training stats
        print('[%d/%d][%d/%d][%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (epoch, num_epochs, i, len(self.dataloader), iters,
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        # Visualization with visdom
        viz.line([errD.item()], [iters], win='ErrorD',
                                   update='append' if iters>0 else None,
                                   name='Loss Discriminator',
                                   opts=dict(
                                       xlabel='Step',
                                       ylabel='Loss',
                                       title='Losses'))
        # Visualization with visdom
        viz.line([errG.item()], [iters], win='ErrorD',
                 update='append' if iters > 0 else None,
                 name='Loss Generator',
                 opts=dict(
                     xlabel='Step',
                     ylabel='Loss',
                     title='Losses'))

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 100 == 0) or ((epoch == num_epochs - 1) and (i == len(self.dataloader) - 1)):
            with torch.no_grad():
                fake = self.netG(self.fixed_noise).detach().cpu()
                viz.images((fake * 0.5 + 0.5).clamp(min=0, max=1), win='generated images')

if __name__=='__main__':
    pokegan = WPokeGan()
    pokegan.train()