import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import argparse
import random
import os
import time
from termcolor import colored
from scipy.io import loadmat
import sys
from networks import DenseNet  ## otherwise define the MLP architecture
import matplotlib.pyplot as plt
import urllib.request
from scipy.interpolate import griddata


class DeepONet(nn.Module):
    # may need to reconsider structure if performance is not ideal
    def __init__(self, branch_net1, branch_net2, trunk_net):
        super().__init__()

        self.branch_net1 = branch_net1
        self.branch_net2 = branch_net2
        self.trunk_net = trunk_net

    def forward(self, branch_inputs1, branch_inputs2, trunk_inputs):

        branch_outputs1 = self.branch_net1(branch_inputs1)
        branch_outputs2 = self.branch_net1(branch_inputs2)
        branch_outputs = branch_outputs1+branch_outputs2

        trunk_outputs = self.trunk_net(trunk_inputs)
        results = torch.einsum('ik, lk -> il', branch_outputs, trunk_outputs)
        return results

def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

parser = argparse.ArgumentParser(description="DeepONet for Linear Darcy Problem in 2D")
# Add arguments for hyperparameters
parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for training')
args = parser.parse_args()
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)

# load and process data
train_data_path = 'linearDarcy_train.mat'
test_data_path = 'linearDarcy_test.mat'
train_data_url = 'https://drive.usercontent.google.com/download?id=1dSjOjI-DHhmUgcFonAqLIqvQIfKouqZo&export=download&authuser=0&confirm=t&uuid=90fa75a0-f578-4d6c-856e-99081265b1d3&at=APZUnTVQ7tBgzAs96tmlIzyl9Z9u:1723137673435'  # Replace with actual URL
test_data_url = 'https://drive.usercontent.google.com/download?id=1_6s-fPzTZCqpysLhfocm6qth8OBl1H-k&export=download&authuser=0&confirm=t&uuid=1a727246-2e09-4f82-a65c-fb2234f105b1&at=APZUnTVNqcWgyHb2nmhkk0jydRL9:1723137701171'   # Replace with actual URL

# Function to download data
def download_data(url, filename):
    print(f'Downloading {filename} from {url}...')
    urllib.request.urlretrieve(url, filename)
    print(f'{filename} downloaded.')

# Check if files exist, otherwise download them
if not os.path.exists(train_data_path):
    download_data(train_data_url, train_data_path)

if not os.path.exists(test_data_path):
    download_data(test_data_url, test_data_path)

data_train = loadmat(train_data_path)
data_test = loadmat(test_data_path)

f1_train = torch.from_numpy(data_train['f1_train']).to(device).float()   
f1_train = f1_train.reshape(f1_train.shape[0], -1)      # 51000 by 961
f2_train = torch.from_numpy(data_train['f2_train']).to(device).float()
f2_train = f2_train.reshape(f2_train.shape[0], -1)
u_train = torch.from_numpy(data_train['u_train']).to(device).float()   # 51000 by 450
x = torch.from_numpy(data_train['x']).to(device).float()
y = torch.from_numpy(data_train['y']).to(device).float()
x_train = torch.cat((x, y), dim=1)     # 450 by 2

f1_test = torch.from_numpy(data_test['f1_test']).to(device).float()
f1_test = f1_test.reshape(f1_test.shape[0], -1)
f2_test = torch.from_numpy(data_test['f2_test']).to(device).float()
f2_test = f2_test.reshape(f2_test.shape[0], -1)
u_test = torch.from_numpy(data_test['u_test']).to(device).float()
x = torch.from_numpy(data_test['x']).to(device).float()
y = torch.from_numpy(data_test['y']).to(device).float()
x_test = torch.cat((x, y), dim=1)

# define and set up model
sensor_size = f1_train.shape[1]
branch_net1 = DenseNet(layersizes=[sensor_size] + [128]*3 + [64], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
branch_net2 = DenseNet(layersizes=[sensor_size] + [128]*3 + [64], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
branch_net1.to(device)
branch_net2.to(device)
trunk_net = DenseNet(layersizes=[2] + [128]*3 + [64], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
trunk_net.to(device)
model = DeepONet(branch_net1, branch_net2, trunk_net)
model.to(device)


# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
train_losses = []
test_losses = []

for epoch in range(epochs):
    permutation = torch.randperm(u_train.size(0))
    for i in range(0, u_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_f1, batch_f2, batch_u, batch_x = f1_train[indices], f2_train[indices], u_train[indices], x_train

        optimizer.zero_grad()
        predictions = model(batch_f1, batch_f2, batch_x)
        loss = criterion(predictions, batch_u)
        loss.backward()
        optimizer.step()
    # Track training loss
    train_losses.append(loss.item())
    #eval
    model.eval()
    with torch.no_grad():
        predictions_test = model(f1_test, f2_test, x_test)
        test_loss = criterion(predictions_test, u_test)
        test_losses.append(test_loss.item())
    model.train()

    if (epoch+1) % (epochs/20) == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}')

# Save model
torch.save(model.state_dict(), 'deeponet_model.pth')
    


# plotting one instance from the test set
predictions_test = model(f1_test, f2_test, x_test)
f1 = f1_test.cpu().numpy()[0,:].reshape(31,31)
f2 = f2_test.cpu().numpy()[0,:].reshape(31,31)
u = u_test.cpu().numpy()[0,:]
u_hat = predictions_test.detach().cpu().numpy()[0,:]
abs_error = np.abs(u - u_hat)

x = x_test.cpu().numpy()[:,0]
y = x_test.cpu().numpy()[:,1]

# Create a figure with 5 subplots
fig, axs = plt.subplots(1, 5, figsize=(16, 3))

im1 = axs[0].imshow(f1, cmap='jet', origin='lower', extent=[0, 1, 0, 1], interpolation='bilinear')
axs[0].add_patch(plt.Rectangle((16/31, 16/31), 15/31, 15/31, fill=True, color='white'))
axs[0].set_title(r'$f_1$')

im2 = axs[1].imshow(f2, cmap='jet', origin='lower', extent=[0, 1, 0, 1], interpolation='bilinear')
axs[1].add_patch(plt.Rectangle((16/31, 16/31), 15/31, 15/31, fill=True, color='white'))
axs[1].set_title(r'$f_2$')

xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
vmin = min(u.min(),u_hat.min())
vmax = max(u.max(),u_hat.max())
grid_x, grid_y = np.mgrid[xmin:xmax:1j*200, ymin:ymax:1j*200]
points = np.array([x.flatten(), y.flatten()]).T
grid_intensity = griddata(points, u, (grid_x, grid_y), method='cubic')
im3 = axs[2].pcolormesh(grid_x, grid_y, grid_intensity, cmap='jet', shading='auto', vmin=vmin,vmax=vmax)
axs[2].add_patch(plt.Rectangle((16/31, 16/31), 15/31, 15/31, fill=True, color='white'))
axs[2].set_title(r'$u$')

grid_intensity_hat = griddata(points, u_hat, (grid_x, grid_y), method='cubic')
im4 = axs[3].pcolormesh(grid_x, grid_y, grid_intensity_hat, cmap='jet', shading='auto', vmin=vmin,vmax=vmax)
axs[3].add_patch(plt.Rectangle((16/31, 16/31), 15/31, 15/31, fill=True, color='white'))
axs[3].set_title(r'$\hat{u}$')

grid_intensity_error = griddata(points, abs_error, (grid_x, grid_y), method='cubic')
im5 = axs[4].pcolormesh(grid_x, grid_y, grid_intensity_error, cmap='jet', shading='auto')
axs[4].add_patch(plt.Rectangle((16/31, 16/31), 15/31, 15/31, fill=True, color='white'))
axs[4].set_title(r'$abs error$')

cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im5, cax=cbar_ax)

plt.tight_layout()
plt.savefig('Darcy_2D_results.png', dpi=400, bbox_inches='tight')