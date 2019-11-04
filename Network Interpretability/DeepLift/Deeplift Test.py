import torch

#%% Define input values
i1 = torch.tensor([0.],requires_grad=True)
i2 = torch.tensor([0.9],requires_grad=True)

#%% Test interpretation sans saturation
y_a = 3*i1+i2

y_a.backward()

print('Grads :\n \ti1 : {} \n \ti2 : {}'.format(i1.grad, i2.grad))
i2.grad.data.zero_();
i1.grad.data.zero_();


#%% Test interpretation avec saturation
h = torch.relu(1-i1-i2)

y = 1-h
y.backward()

print('Grads :\n \ti1 : {} \n \ti2 : {}'.format(i1.grad, i2.grad))
i2.grad.data.zero_();
i1.grad.data.zero_();
