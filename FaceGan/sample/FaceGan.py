from sample.settings import *
import torch.nn as nn
from sample.helpers import load_dataset,weights_init
from sample.Discriminator import Discriminator
from sample.Generator import Generator
import torch.optim as optim

class FaceGan() :

    def __init__(self):
        self.dataset, self.dataloader, self.device = load_dataset()

        self.netG = Generator(ngpu).to(self.device)
        self.netD = Discriminator(ngpu).to(self.device)

        # Initialise Weights
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)


        # define loss function
        self.criterion = nn.BCELoss()

        # We create a fixed subset of random for the latent variable, this way we can evaluate our progress.
        self.fixed_noise = torch.randn(n_visualisation, nz, 1, 1, device=self.device)

        # Establish convention for real and fake labels during training
        self.real_label = 1
        self.fake_label = 0

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))


        # Fixed noise for visualisation
        self.fixed_noise = torch.randn(64, nz, 1, 1, device=self.device)

    def train(self):
        # Training Loop

        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                ## Reset the grad for the Discriminator Network
                self.netD.zero_grad()

                # Format batch
                real_cpu = data[0].to(self.device)  # Load the batch to gpu

                # Create the vector of the input size with only True = [real_label]*b_size
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), self.real_label, device=self.device)

                # Forward pass real batch through D -> Get predictions
                output = self.netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, label)

                # Calculate gradients for D in backward pass, WAIT to optimize !!!
                errD_real.backward()
                D_x = output.mean().item()  # Accuracy of the classifier for True examples

                # Train with all-fake batch
                # Generate batch of latent vectors using noise, same number of noise than real examples
                noise = torch.randn(b_size, nz, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.netG(noise)
                label.fill_(self.fake_label)  # Convert the previously label vector to all False this time
                # Classify all fake batch with D, have to detach so the optimization don't impact netG !
                output = self.netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()  # 1-Accuracy of the classifier for generated Examples
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # We can now optimize netD
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(self.real_label)  # Here we use fake images

                output = self.netD(fake).view(-1)  # We use the same fake because generator didn't changed yet
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G , We only update
                errG.backward()  # loos for G
                D_G_z2 = output.mean().item()  # G sucess faking
                # Update G
                self.optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, i, len(self.dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(self.dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.netG(self.fixed_noise).detach().cpu()
                        viz.images((fake * 0.5 + 0.5).clamp(min=0, max=1),win='generated images')

                iters += 1

if __name__=='__main__':
    faceGan = FaceGan()
    faceGan.train()