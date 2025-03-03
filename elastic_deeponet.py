from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import os
import sys
import random
import urllib.request
from scipy.io import loadmat
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata


import argparse
from tqdm import trange
# from src.DeepONet import DeepONet
# from src.Datasets.ElasticPlateDataset import ElasticPlateBoudaryForceDataset, ElasticPlateDisplacementDataset,plot_target_boundary, plot_source_boundary_force, plot_transformation_elastic

sys.path.append("../..")
sys.path.append('..')
sys.path.append('../operator_testing')
from operator_testing.deeponet_derivative import KANBranchNet, KANTrunkNet
from networks import *
import efficient_kan
# from efficient_kan import *
import kan

class DeepONet(nn.Module):
    def __init__(self, branch_net, trunk_net):
        super().__init__()

        self.branch_net = branch_net
        self.trunk_net = trunk_net

    def forward(self, branch_inputs, trunk_inputs):
        # print(f"Branch input shape: {branch_inputs.shape}")
        branch_inputs = branch_inputs.reshape(branch_inputs.shape[0], -1)
        branch_outputs = self.branch_net(branch_inputs)
        branch_outputs = branch_outputs.reshape(branch_outputs.shape[0], -1, 2)
        # print(f"Branch output shape: {branch_outputs.shape}")

        # print(f"Trunk input shape: {trunk_inputs.shape}")
        # trunk_inputs = trunk_inputs.reshape(trunk_inputs.shape[0], -1)
        trunk_outputs = self.trunk_net(trunk_inputs)
        trunk_outputs = trunk_outputs.reshape(trunk_outputs.shape[0], trunk_inputs.shape[1], -1, 2)
        # print(f"Trunk output shape: {trunk_outputs.shape}")
        results = torch.einsum('fpz, fdpz -> fdz', branch_outputs, trunk_outputs)
        return results

def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="DeepONet for Elastic Plate problem.")
    # Add arguments for hyperparameters
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--modeltype', type=str, default='densenet', help='Branch/Trunk model.',
                        choices=['densenet', 'efficient_kan'])
    parser.add_argument('-mode', dest='mode', type=str, default='shallow',
                            help='Network architecture mode.',
                            choices=['shallow', 'deep'])
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    modeltype = args.modeltype
    mode = args.mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)

    seed=0
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    #defining the output directory
    output_dir = os.path.join(os.getcwd(), f'ElasticPlate_DeepONet/{mode}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load and process data
    train_data_path = 'linearElasticity_train.mat'
    test_data_path = 'linearElasticity_test.mat'
    train_data_url = "https://drive.usercontent.google.com/download?id=1Am7TLUFEWQ6rWviJB-V0NDJoOdhkbUPn&export=download&authuser=0&confirm=t&uuid=d22af85e-e7f2-4186-84a6-0b7714aa60df&at=APZUnTUIfZKE27uCb0gWN0VxOWb8:1723137813630"
    test_data_url = "https://drive.usercontent.google.com/download?id=1nXnZm-2MKpnH22CJ7bRmLTqA0WC9cxG8&export=download&authuser=0&confirm=t&uuid=2924183e-b5da-4085-8531-75f77ca81333&at=APZUnTUeqXTPa2Jpseg7X44P4iQP:1723137855875"

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

    print("Train data keys: ", data_train.keys())

    #NOTE: based on tyler's code, the DeepONet will not use the xs (concat of xx and yy).
    #The model uses u and uy to define the ys.
    xx = data_train['x']
    yy = data_train['y']
    ux = data_train['ux_train']
    uy = data_train['uy_train']
    f_train = torch.from_numpy(data_train['f_train']).to(device).float()

    branch_inputs_train = f_train
    ys = np.concatenate((ux[:, :, None], uy[:, :, None]), axis=2)
    
    uxs = np.concatenate((xx,yy),axis=1)
    uxs = np.repeat(uxs[None,:,:], ys.shape[0], axis=0)
    uxs = torch.tensor(uxs).to(device).float()
    trunk_inputs_train = uxs

    std = 4.263603113940917e-05
    ys = torch.tensor(ys).to(device).float()/std
    outputs_train = ys

    print(f"Branch inputs shape: {branch_inputs_train.shape}")
    print(f"Trunk inputs shape: {trunk_inputs_train.shape}")
    print(f"Targets shape: {outputs_train.shape}")

    #Repeat the above for the test data.
    print("Test data keys: ", data_test.keys())
    xx_test = data_test['x']
    yy_test = data_test['y']
    ux_test = data_test['ux_test']
    uy_test = data_test['uy_test']
    f_test = torch.from_numpy(data_test['f_test']).to(device).float()

    branch_inputs_test = f_test
    ys_test = np.concatenate((ux_test[:, :, None], uy_test[:, :, None]), axis=2)
    
    uxs_test = np.concatenate((xx_test,yy_test),axis=1)
    uxs_test = np.repeat(uxs_test[None,:,:], ys_test.shape[0], axis=0)
    uxs_test = torch.tensor(uxs_test).to(device).float()
    trunk_inputs_test = uxs_test

    std = 4.263603113940917e-05
    ys_test = torch.tensor(ys_test).to(device).float()/std
    outputs_test = ys_test

    print(f"TEST branch inputs shape: {branch_inputs_test.shape}")
    print(f"TEST trunk inputs shape: {trunk_inputs_test.shape}")
    print(f"TEST targets shape: {outputs_test.shape}")

    # Here we define and set up model. 
    # Note from Tyler's code: branch is 2*input_sensors, hidden, and then 2*p.
    # Trunk is 2, hidden, and then 2*p.
    input_sensors = f_train.shape[1]
    input_neurons_branch = input_sensors
    input_neurons_trunk = 2
    p = 100 #standardized across operator learning problems.

    if mode == 'deep':
        if modeltype=='efficient_kan':
            # branch_net = efficient_kan.KAN(layers_hidden=[input_neurons_branch] + [2*input_neurons_branch+1]*1 + [p])
            # trunk_net = efficient_kan.KAN(layers_hidden=[input_neurons_trunk] + [2*input_neurons_trunk+1]*1 + [p])
            branch_net = efficient_kan.KAN(layers_hidden=[input_neurons_branch]+[100]*3+[p])
            branch_net.to(device)
            trunk_net = efficient_kan.KAN(layers_hidden=[input_neurons_trunk]+[100]*3+[p])
            trunk_net.to(device)
        else:
            branch_net= DenseNet(layersizes=[input_neurons_branch] + [256]*3 + [p], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
            branch_net.to(device)
            trunk_net = DenseNet(layersizes=[input_neurons_trunk] + [256]*3 + [p], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
            trunk_net.to(device)
    elif mode == 'shallow':
        if modeltype=='efficient_kan':
            branch_net = efficient_kan.KAN(layers_hidden=[input_neurons_branch] + [2*input_neurons_branch+1]*1 + [p])
            trunk_net = efficient_kan.KAN(layers_hidden=[input_neurons_trunk] + [2*input_neurons_trunk+1]*1 + [p])
            # branch_net1 = efficient_kan.KAN(layers_hidden=[input_neurons_branch]+[2*input_neurons_branch]*2+[p])
            # branch_net2 = efficient_kan.KAN(layers_hidden=[input_neurons_branch]+[2*input_neurons_branch]*2+[p])
            branch_net.to(device)
            # trunk_net = efficient_kan.KAN(layers_hidden=[input_neurons_trunk]+[2*input_neurons_trunk]*2+[p])
            trunk_net.to(device)
        else:
            branch_net = DenseNet(layersizes=[input_neurons_branch] + [1000] + [p], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
            branch_net.to(device)
            trunk_net = DenseNet(layersizes=[input_neurons_trunk] + [1000] + [p], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
            trunk_net.to(device)
    else:
        print('Incorrect architecture mode provided, must be one of "shallow" or "deep".')
        return
    model = DeepONet(branch_net, trunk_net)
    model.to(device)
    print(f"Running ELASTIC PLATE with modeltype {modeltype}, architecture mode {mode}.")


    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-4)
    criterion = nn.MSELoss()
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        permutation = torch.randperm(f_train.size(0))
        for i in range(0, f_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_ux, batch_y, batch_tgt = branch_inputs_train[indices], trunk_inputs_train[indices], outputs_train[indices]

            optimizer.zero_grad()
            predictions = model(batch_ux, batch_y)
            loss = criterion(predictions, batch_tgt)
            loss.backward()
            optimizer.step()
        # scheduler.step()
        # Track training loss
        train_losses.append(loss.item())
        #eval
        model.eval()
        with torch.no_grad():
            predictions_test = model(branch_inputs_test, trunk_inputs_test)
            test_loss = criterion(predictions_test, outputs_test)
            test_losses.append(test_loss.item())
        model.train()

        if (epoch+1) % (epochs/50) == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

    # Save model
    torch.save(model.state_dict(), f'{output_dir}/{modeltype}_deeponet_model.pt')
    np.save(f'{output_dir}/{modeltype}_deeponet_model_loss_list.npy', np.asarray(train_losses))
    np.save(f'{output_dir}/{modeltype}_deeponet_model_test_loss_list.npy', np.asarray(test_losses))

    #plotting the model training and test losses.
    plt.figure()
    plt.plot(np.arange(epochs), train_losses, label='Training Loss')
    plt.plot(np.arange(epochs), test_losses, label='Test Loss')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    # plt.yticks(ticks=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3])
    plt.tight_layout()
    plt.title(f'{modeltype} Elastic Plate Train/Test Losses')
    plt.savefig(f'{output_dir}/{modeltype}_losses.jpg')
        


    # Taken from Tyler's code: plots the target boundary.
    def plot_target_boundary(xs, ys, y_hats, modeltype, mode):

        # 4 rows, 5 cols. Last col is only 0.2 wide as it is a colorbar.
        # fig, axs = plt.subplots(4, 4, figsize=(20, 20))
        fig = plt.figure(figsize=(4.4 * 5, 4.5 * 4), dpi=300)
        gridspec = fig.add_gridspec(4, 5, width_ratios=[1, 1, 1, 1, 0.2])

        # v_min and vmax based on data
        vmin = ys[0:4].min().item()
        vmax = ys[0:4].max().item()

        # each row is 1 function
        # first 2 cols are groundtruth x,y displacements
        # last 2 cols are estimated x,y displacements

        for row in range(4):

            # get input
            xx, yy = xs[row, :, 0].cpu(), xs[row, :, 1].cpu()

            # get output data
            groundtruth_displacement_x = ys[row, :, 0]
            groundtruth_displacement_y = ys[row, :, 1]
            predicted_displacement_x = y_hats[row, :, 0]
            predicted_displacement_y = y_hats[row, :, 1]


            # plot details
            image_density = 200j
            grid_x, grid_y = np.mgrid[-xx.min().cpu().item():xx.max().cpu().item():image_density, 
                                    -yy.min().cpu().item():yy.max().cpu().item():image_density]
            points = np.array([xx.flatten().cpu(), yy.flatten().cpu()]).T

            for col in range(4):
                intensity = groundtruth_displacement_x if col == 0 else \
                            groundtruth_displacement_y if col == 1 else \
                            predicted_displacement_x if col == 2 else \
                            predicted_displacement_y
                grid_intensity = griddata(points, intensity.cpu(), (grid_x, grid_y), method='cubic')

                # remove the points inside the hole
                max_distance = (xx.max() - xx.min()) / 10
                for i in range(grid_x.shape[0]):
                    for j in range(grid_x.shape[1]):
                        distance = np.sqrt((grid_x[i, j] - xx) ** 2 + (grid_y[i, j] - yy) ** 2).min()
                        if distance > max_distance:
                            grid_intensity[i, j] = np.nan

                # mesh plot
                ax = fig.add_subplot(gridspec[row, col])
                mesh = ax.pcolormesh(grid_x, grid_y, grid_intensity, cmap='jet', shading='auto')

                # save
                plt.gca().set_aspect('equal', adjustable='box')
                plt.xlabel("x")
                plt.ylabel("y")
                title = "Groundtruth x displacement" if col == 0 else \
                        "Groundtruth y displacement" if col == 1 else \
                        "Predicted x displacement" if col == 2 else \
                        "Predicted y displacement"
                ax.set_title(title)

        # add color bar
        ax = fig.add_subplot(gridspec[:, 4])
        cbar = plt.colorbar(mesh, cax=ax)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/target_{modeltype}_{mode}.png")

    model.eval()
    preds = model(branch_inputs_test, trunk_inputs_test).detach().cpu().numpy()
    outputs_test = outputs_test.cpu().numpy()
    l2_errors = np.linalg.norm(preds - outputs_test, axis=1) / np.linalg.norm(outputs_test, axis=1)
    print(f"Mean L2 error: {np.mean(l2_errors)}")
    np.save(f'{output_dir}/{modeltype}_l2_errs', np.asarray(l2_errors))
    np.save(f'{output_dir}/{modeltype}_preds', np.asarray(preds))
    plot_target_boundary(trunk_inputs_test, outputs_test, model(branch_inputs_test, trunk_inputs_test), modeltype, mode)

    return

if __name__ == '__main__':
    main()