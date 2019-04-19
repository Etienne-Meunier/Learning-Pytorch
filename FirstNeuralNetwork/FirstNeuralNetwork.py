import torch.nn as nn
import torch.optim as optim
from FirstNeuralNetwork.loading_blocks import load_preprocess_datas, write_submission
import torch
import numpy as np


class FirstNeuralNet(nn.Module):

    def __init__(self, n_input):
        """
        In that function we define the different layers of the neural network
        n_input : number of dimension of the input data
        """
        super(FirstNeuralNet, self).__init__()  # call the super constructor from Pytorch nn.Module

        # Only 2 fully connected layers, we have one as final output because we want the price, so a scalar
        self.fc1 = nn.Linear(n_input, 10)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(10, 5)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(5, 1)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        """
        We define the forward loop, so all the steps that will take the neural network
        x : input of the forward loop, so the features to predict, of shape [n_batch,n_input_shape]
        return : output of the feed forward, so y_hat
        """
        x = self.fc1(x).clamp(min=0)
        x = self.fc2(x).clamp(min=0) # Same as relu activation
        x = self.fc3(x)  # Execution of the Second layer + Linear activation because it's a regression
        return x


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, X_submit, index_submit = load_preprocess_datas()

    net = FirstNeuralNet(X_train.shape[1])

    # Loss Function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(net.parameters())

    n_iterations = 100000

    last_loss_test = 0

    # Typlical training loop:
    for i in range(n_iterations):
        output = net(X_train)
        loss = torch.sqrt(criterion(torch.log1p(output), torch.log1p(y_train)))

        if i % 1000 == 0:
            output_test = net(X_test)
            loss_test = torch.sqrt(criterion(torch.log1p(output_test), torch.log1p(y_test)))
            print('Iteration {} Loss Train : {} Loss Test {}'.format(i, loss, loss_test))
            if np.abs(last_loss_test - loss_test.item()) < 0.001 :
                break
            last_loss_test= loss_test.item()

        optimizer.zero_grad()  # zero the gradient buffers
        loss.backward()
        optimizer.step()  # Does the update

    y_submit = net(X_submit)
    write_submission(y_submit.data.numpy(), index_submit, file_name='submission.csv')

    '''
    # print parameters :
    #print(list(net.parameters()))
    # Get an output
    output = net(torch.randn(1,n_input))
    # Zero out the gradient history - Reset the net
    net.zero_grad()
    # Define a loss function to evaluate our network and test with random target
    criterion = torch.nn.MSELoss()
    loss = criterion(output,torch.rand(1))
    # Do backpropagation : all we have to do is :
    loss.backward()
    # Update the weights using Stochastic Gradient Descent

    #1. We could use python ( and that is cool )
    learning_rate = 0.01
    for f in net.parameters():
        f.data.sub_(f.grad.data * learning_rate)

    #2. We could also use the given methods by Pytorch ( it automatically uses the gradient stored)
    optimizer = optim.SGD(net.parameters(),lr=learning_rate)
    optimizer.step()
    '''
