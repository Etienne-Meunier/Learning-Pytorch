from settings import *
import torch.nn as nn
from helpers import load_dataset,weights_init,display_city,create_viz,create_logger
from Discriminator import Discriminator
from Generator import Generator
from natsort import natsorted
import torch.optim as optim
import argparse
import logging
from datetime import datetime
import os


class MicroGan() :

    def __init__(self,experiment_name, vizualize,num_epochs,n_observations):
        # Create Experiment name dir for records
        self.experiment_name = experiment_name
        self.n_observations = n_observations

        self.viz = create_viz('{}_{}'.format(name_env,self.experiment_name)) if vizualize else None

        self.dataset, self.dataloader, self.device = load_dataset(self.viz, folder_name=self.experiment_name)

        self.netG = Generator(ngpu).to(self.device)
        self.netD = Discriminator(ngpu).to(self.device)

        self.start_epoch = self.filehandling_experiment()
        self.num_epochs = num_epochs



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
        epoch =self.start_epoch + 1
        for e in range(self.num_epochs):
            epoch = self.start_epoch + e + 1
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
                G1 = torch.sigmoid(fake_predictions).mean()
                D = torch.sigmoid(real_predictions).mean()

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
                G2 = torch.sigmoid(fake_predictions).mean()
                # loss Generator = -mean(prediction fake)
                loss_generator = -fake_predictions.mean()
                # zero out the grad for Generator
                self.netG.zero_grad()
                # Backward la loss Generator
                loss_generator.backward()
                # OptimizerG
                self.optimizerG.step()
            self.visualize_progession(epoch, loss_discriminator.item(), loss_generator.item(),G1 ,D)
            self.save_model(epoch, loss_discriminator.item(), loss_generator.item())

        self.save_observations(epoch)

    def visualize_progession(self, epoch, loss_discriminator, loss_generator, G1, D):
        """
        Handle the printing and plotting of the loss
        :return:
        """
        # Output training stats
        logging.info('[{}/{}]\tLoss_D: {:.4f}\tLoss_G: {:.4f} %Fake as Real: {:.1f} %Real as real: {:.1f}'.format(epoch,self.num_epochs+self.start_epoch, loss_discriminator, loss_generator, G1*100,D*100))
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
                    display_city(fake[0],win_name='Test Example',viz=self.viz,folder_name=self.experiment_name,epoch=epoch)


    def save_observations(self,epoch):
        """
        Save the n_observations generated cities for a given layer
        :param n_observations: number of observations to save
        """
        logging.info('Save {} observations at epoch {}'.format(self.n_observations,epoch))
        with torch.no_grad():
            fake = self.netG(self.fixed_noise).detach().cpu()
            for i in range(self.n_observations) :
                display_city(fake[i], win_name='Test Example', folder_name=self.experiment_name, epoch='observation_{}_epoch_{}'.format(i, epoch))

    def save_model(self,epoch,loss_discriminator,loss_generator):
        if epoch%30 == 0 or epoch == self.start_epoch+self.num_epochs:
            logging.info('Save model epoch {}'.format(epoch))
            torch.save({
                'epoch': epoch,
                'netD_state_dict': self.netD.state_dict(),
                'netG_state_dict' : self.netG.state_dict(),
                'errD': loss_discriminator,
                'errG': loss_generator
            }, '{}{}/models/model_epoch_{}.tar'.format(path_experiments,self.experiment_name,epoch))

    def filehandling_experiment(self):
        """
        Load the model of the experiment if it exist or create the file structure to save it if note
        :return: Number of epochs of the model loaded, 0 if the experiment just created
        """
        if os.path.isdir("{}{}".format(path_experiments, self.experiment_name)):
            checkpoints = natsorted(os.listdir("{}{}/models".format(path_experiments, self.experiment_name)))# sort the checkpoints to get the last one
            print('Load Experiment {} Model : {}'.format(self.experiment_name,checkpoints[-1]))
            checkpoint = torch.load("{}{}/models/{}".format(path_experiments,self.experiment_name,checkpoints[-1])) # Load the last model
            self.netD.load_state_dict(checkpoint['netD_state_dict'])
            self.netG.load_state_dict(checkpoint['netG_state_dict'])
            start_epoch = checkpoint['epoch']
        else:
            print('Create Experiment {}'.format(self.experiment_name))
            os.mkdir("{}{}".format(path_experiments, self.experiment_name))
            os.mkdir("{}{}/outputs".format(path_experiments, self.experiment_name))
            os.mkdir("{}{}/models".format(path_experiments, self.experiment_name))
            start_epoch = 0

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename='{}{}/logs_output.log'.format(path_experiments, self.experiment_name), level=logging.INFO)
        return start_epoch


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualisation", help="Activate Visdom visualization", default=False,type=bool)
    parser.add_argument("--epochs", help="Number of epochs for training", default=100,type=int)
    parser.add_argument("--experiment", help="Name of the experiment")
    parser.add_argument("--n_observations", help="Number of generated object to save at the end of the training",default=10,type=int)
    args = parser.parse_args()
    microgan = MicroGan(experiment_name=args.experiment, vizualize=args.visualisation, num_epochs=args.epochs, n_observations=args.n_observations)
    microgan.train()