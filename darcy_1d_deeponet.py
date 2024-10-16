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
from networks import DenseNet  ## otherwise defined the MLP 
import matplotlib.pyplot as plt
import urllib.request


class DeepONet(nn.Module):

    def __init__(self, branch_net, trunk_net):
        super().__init__()
        
        self.branch_net = branch_net
        self.trunk_net = trunk_net
    
    def forward(self, branch_inputs, trunk_inputs):
        
        branch_outputs = self.branch_net(branch_inputs)
        trunk_outputs = self.trunk_net(trunk_inputs)    
        results = torch.einsum('ik, lk -> il', branch_outputs, trunk_outputs)   
        return results

def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)

# load and process data
train_data_path = 'nonlineardarcy_train.mat'
test_data_path = 'nonlineardarcy_test.mat'
train_data_url = 'https://drive.usercontent.google.com/download?id=1DTAkXxMjB2xekkqQt1CLwGeKtz5LP6e1&export=download&authuser=0&confirm=t&uuid=cbe6a789-0751-466b-a95d-ee33ff43d6c1&at=APZUnTVEUevchEMmdzxatZeZ3Saj:1723137744388'  # Replace with actual URL
test_data_url = 'https://drive.usercontent.google.com/download?id=1PsKE5yQkIM8fWBdC51d9VKP55TQZ06OC&export=download&authuser=0&confirm=t&uuid=51001974-28a4-468b-a717-20dc3e80a98e&at=APZUnTVp9b1TpDyRuapd8isoC-WR:1723137779616'   # Replace with actual URL

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

input_train = torch.from_numpy(data_train['f_train']).to(device).float()
output_train = torch.from_numpy(data_train['u_train']).to(device).float()
x_train = torch.from_numpy(data_train['x']).to(device).float().t()

input_test = torch.from_numpy(data_test['f_test']).to(device).float()
output_test = torch.from_numpy(data_test['u_test']).to(device).float()
x_test = torch.from_numpy(data_test['x']).to(device).float().t()

branch_net = DenseNet(layersizes=[50] + [50]*3 + [20], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
branch_net.to(device)
trunk_net = DenseNet(layersizes=[1] + [50]*3 + [20], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
trunk_net.to(device)
model = DeepONet(branch_net, trunk_net)
model.to(device)



# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training loop
epochs = 1000
batch_size = 32
train_losses = []

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(input_train.size(0))

    for i in range(0, input_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_input, batch_output, batch_x = input_train[indices], output_train[indices], x_train
        
        optimizer.zero_grad()
        predictions = model(batch_input, batch_x)
        loss = criterion(predictions, batch_output)
        loss.backward()
        optimizer.step()

    # Track training loss
    train_losses.append(loss.item())
    if (epoch+1) % (epochs/10) == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

# Evaluation
model.eval()
with torch.no_grad():
    predictions_test = model(input_test, x_test)
    test_loss = criterion(predictions_test, output_test)
    print(f'Test Loss: {test_loss.item():.6f}')
    
    # Calculate additional metrics
    rmse = torch.sqrt(test_loss).item()
    print(f'Test RMSE: {rmse:.6f}')

# Save model
torch.save(model.state_dict(), 'deeponet_model.pth')



def plot_predictions(predictions, ground_truth, x, num_instances=3):
    plt.figure(figsize=(12, 6))
    
    for i in range(num_instances):
        plt.subplot(1, num_instances, i + 1)
        plt.plot(x.cpu().numpy(), ground_truth[i].cpu().numpy(), label='Ground Truth', color='blue')
        plt.plot(x.cpu().numpy(), predictions[i].cpu().numpy(), label='Prediction', color='red', linestyle='dashed')
        plt.title(f'Instance {i+1}')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('predictions_and_true.png', dpi=400,bbox_inches='tight')
    plt.show()

# Evaluation and plotting
model.eval()
with torch.no_grad():
    predictions_test = model(input_test, x_test)  
    # Plot the first 3 instances
    plot_predictions(predictions_test, output_test, x_test, num_instances=3)