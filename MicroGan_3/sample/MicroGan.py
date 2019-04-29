from settings import *
import torch.nn as nn
from helpers import load_dataset,weights_init,display_city,create_viz
from Discriminator import Discriminator
from Generator import Generator
import torch.optim as optim

class MicroGan() :

    def __init__(self,vizualize):
        self.viz = create_viz(name_env) if vizualize else None
        self.dataset, self.dataloader, self.device = load_dataset(self.viz)

        self.netG = Generator(ngpu).to(self.device)
        self.netD = Discriminator(ngpu).to(self.device)

        # Initialise Weights
        #self.netG.apply(weights_init)
        #self.netD.apply(weights_init)

        # We create a fixed subset of random for the latent variable, this way we can evauate our progress.
        self.fixed_noise = torch.randn(64, nz, 1, 1, device=self.device)

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.RMSprop(self.netD.parameters(), lr=lr)
        self.optimizerG = optim.RMSprop(self.netG.parameters(), lr=lr)


        # Fixed noise for visualisation
        self.fixed_noise = torch.randn(64, nz, 1, 1, device=self.device)

    def train(self):
        # Training Loop


        i_dataloader = iter(self.dataloader)

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            for d in range(d_iters) :
                # Load real data from dataloader and load in cpu + compute batch size for this one
                try:
                    real_images, _ = next(i_dataloader)
                except StopIteration:
                    i_dataloader = iter(self.dataloader)
                    real_images, _ = next(i_dataloader)
                real_images_cpu = real_images.to(self.device)
                b_size = real_images_cpu.size(0)
                # compute predictions for real images
                real_predictions = self.netD(real_images_cpu).view(-1) # Compute predictions and flatten

                # Generate random Serie
                noise = torch.randn(b_size, nz, 1, 1, device=self.device)
                # create fake images
                fake = self.netG(noise)
                # compute predictions for fake images
                fake_predictions = self.netD(fake.detach()).view(-1)


                # loss Discriminator = mean(prediction fake) - mean(prediction real)
                loss_discriminator = -(real_predictions.mean() - fake_predictions.mean())
                G1 = fake_predictions.mean()
                D = real_predictions.mean()

                # zero out the grad for Discriminator
                self.netD.zero_grad()
                # Backward la loss Discriminator
                loss_discriminator.backward()
                # OptimizerD
                self.optimizerD.step()
                # Clip the weights
                for p in self.netD.parameters():
                    p.data.clamp_(-c, c)

            for g in range(g_iters) :
                # Generate random serie
                noise = torch.randn(batch_size, nz, 1, 1, device=self.device)
                # Create fake image
                fake = self.netG(noise)
                # compute prediction for fake images again
                fake_predictions = self.netD(fake).view(-1)
                G2 = fake_predictions.mean()
                # loss Generator = -mean(prediction fake)
                loss_generator = -fake_predictions.mean()
                # zero out the grad for Generator
                self.netG.zero_grad()
                # Backward la loss Generator
                loss_generator.backward()
                # OptimizerG
                self.optimizerG.step()
            self.visualize_progession(epoch, loss_discriminator.item(), loss_generator.item(),G1,G2,D)
            self.save_model(epoch, loss_discriminator.item(), loss_generator.item())

    def visualize_progession(self, epoch, loss_discriminator, loss_generator, G1, G2, D):
        """
        Handle the printing and plotting of the loss
        :return:
        """
        # Output training stats
        print('[{}/{}]\tLoss_D: {:.4f}\tLoss_G: {:.4f} G1: {:.4f} G2: {:.4f} D: {:.4f}'.format(epoch, num_epochs, loss_discriminator, loss_generator, G1,G2,D))
        if self.viz :
            # Visualization with visdom
            self.viz.line([loss_discriminator], [epoch], win='ErrorD',
                                       update='append' if epoch>0 else None,
                                       name='Loss Discriminator',
                                       opts=dict(
                                           xlabel='Step',
                                           ylabel='Loss',
                                           title='Losses'))
            # Visualization with visdom
            self.viz.line([loss_generator], [epoch], win='ErrorD',
                     update='append' if epoch > 0 else None,
                     name='Loss Generator',
                     opts=dict(
                         xlabel='Step',
                         ylabel='Loss',
                         title='Losses'))

            # Check how the generator is doing by saving G's output on fixed_noise
            if (epoch % 10 == 0) :
                with torch.no_grad():
                    fake = self.netG(self.fixed_noise).detach().cpu()
                    display_city(fake[0],win_name='Test Example',viz=self.viz,epoch=epoch)



    def save_model(self,epoch,loss_discriminator,loss_generator):
        if epoch%30 == 0 :
            print('Save model epoch {}'.format(epoch))
            torch.save({
                'epoch': epoch,
                'netD_state_dict': self.netD.state_dict(),
                'netG_state_dict' : self.netG.state_dict(),
                'errD': loss_discriminator,
                'errG': loss_generator
            }, '../models/model__04_24_19.tar')


if __name__=='__main__':
    microgan = MicroGan()
    microgan.train()