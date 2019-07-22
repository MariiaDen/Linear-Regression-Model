import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
import numpy as np

torch.manual_seed(5)

#actual data
X = 10 * torch.randn(100, 1)
Y = X + 4 * torch.randn(100, 1)
Z = Y + 3 * torch.randn(100, 1)

class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        pred = self.linear(x)
        return pred
    
model = LR(2, 1)
features = torch.cat((X, Y), 1)

def get_param():
    [w, b] = model.parameters()
    return(w[0][0].item(), w[0][1].item(), b[0].item())

def plot_all():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    X_fit = torch.linspace(-30, 30, 100)
    Y_fit = torch.linspace(-30, 30, 100)
    w1, w2, b = get_param()
    Z_fit = w1*X_fit + w2*Y_fit + b
    scat = ax.scatter(X, Y, Z)
    scat = ax.plot(X_fit.numpy(), Y_fit.numpy(), Z_fit.squeeze().numpy(), label='parametric curve')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.grid()
    plt.show()

plot_all()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

epochs = 100 
losses = []
for i in range(epochs):
    Z_fit = model.forward(features)
    loss = criterion(Z_fit, Z)
    losses.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(losses)
plot_all()
