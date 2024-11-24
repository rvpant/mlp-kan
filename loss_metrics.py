import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from DeepONet import DeepONet
from networks import DenseNet
import efficient_kan
from scipy.io import loadmat
from darcy_2d_deeponet import DeepONet as Darcy_DeepONet_2D #to distinguish from the  model used in the other two cases.

#PART I: functions for the loss landscape analysis

def get_loss_onet(model, branch_data, trunk_data, targets):
    model.eval()
    preds = model(branch_data, trunk_data)
    # print("Model evaluation shape: ", preds.shape)
    loss_fn = nn.MSELoss()
    loss = loss_fn(preds, targets)
    
    return loss

def get_random_direction(model):
    return [torch.randn_like(p) for p in model.parameters()]

def normalize_direction(direction):
    norm = torch.sqrt(sum(torch.sum(d**2) for d in direction))
    return [d / norm for d in direction]

def update_model(model, direction, step):
    with torch.no_grad():
        for p, d in zip(model.parameters(), direction):
            p.add_(step * d)

def reset_model(model, original_params):
    with torch.no_grad():
        for p, orig_p in zip(model.parameters(), original_params):
            p.copy_(orig_p)

def compute_loss_landscape_onet(model, branch_data, trunk_data, targets, n_points = 20, step_size=0.5):
    original_params = [p.clone() for p in model.parameters()]
    # original_params.to(device)
    dir1 = normalize_direction(get_random_direction(model))
    dir2 = normalize_direction(get_random_direction(model))

    steps = torch.linspace(-step_size, step_size, n_points)
    loss_landscape = torch.zeros((n_points, n_points))
    for i, step1 in enumerate(steps):
        for j, step2 in enumerate(steps):
            update_model(model, dir1, step1)
            update_model(model, dir2, step2)
            loss_landscape[i, j] = get_loss_onet(model, branch_data, trunk_data, targets).item()
            reset_model(model, original_params)

    return loss_landscape

def plot_loss_landscape(loss_landscape, title, modelflag, outdir='.'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.linspace(-1, 1, loss_landscape.shape[0])
    X, Y = np.meshgrid(x, y)

    surf = ax.plot_surface(X, Y, loss_landscape, cmap='viridis')
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Loss')
    ax.set_title(title)
    fig.colorbar(surf)
    plt.show()
    plt.savefig(f'{outdir}/onet_loss_landscapes_{modelflag}.jpg')

    return None

#PART II: Rademacher complexity and VC dimension implementation

def rademacher_complexity(model, branch_data, trunk_data, device, num_samples=1000):
    """
    Estimate the empirical Rademacher complexity of a model.

    Args:
    model (nn.Module): The neural network model
    x (torch.Tensor): Input data
    num_samples (int): Number of Rademacher samples to use

    Returns:
    float: Estimated Rademacher complexity
    """

    n = branch_data.shape[0]

    with torch.no_grad():
        onet_output = model(branch_data, trunk_data); onet_output.to(device)
    # print(n, onet_output, onet_output.shape)
    complexity = 0
    for _ in range(num_samples):
        sigma = torch.randint(0,2,(n,)).float() * 2 - 1
        sigma = sigma.to(device)
        # print(onet_output.get_device())
        # print(sigma.get_device())
        complexity += torch.abs(torch.sum(onet_output.T*sigma)) / n
    
    return complexity/num_samples

def compute_rademacher_curve(model, branch_data, trunk_data, device, max_samples=500, step=10):
    """
    Compute Rademacher complexity curve for increasing sample sizes.

    Args:
    model (nn.Module): The neural network model
    x (torch.Tensor): Input data
    max_samples (int): Maximum number of samples to use
    step (int): Step size for increasing samples

    Returns:
    tuple: Lists of sample sizes and corresponding Rademacher complexities
    """

    sample_sizes = list(range(step, min(len(branch_data), max_samples)+1, step))
    complexities = []
    for size in sample_sizes:
        branch_subset = branch_data[:size]; branch_subset.to(device)
        trunk_subset = trunk_data[:size]; trunk_subset.to(device)
        complexity = rademacher_complexity(model, branch_subset, trunk_subset, device).cpu()
        complexities.append(complexity)
    
    return sample_sizes, complexities

#TODO: modify this for the onet.
def plot_onet_rademacher_curves(sample_sizes, complexities_onet, complexities_base, outdir='.'):
    """
    Plot Rademacher complexity curves the onet model.

    Args:
    sample_sizes (list): List of sample sizes
    complexities_standard (list): Rademacher complexities for standard NN
    complexities_pinn (list): Rademacher complexities for PINN
    """
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, complexities_onet, label='DeepONet - KAN')
    plt.plot(sample_sizes, complexities_base, label='Baseline DenseNet')
    plt.xlabel('Sample Size')
    plt.ylabel('Complexity')
    plt.title('Rademacher Complexity KAN vs DenseNet')
    plt.legend()
    plt.show()  
    plt.savefig(f'{outdir}/onet_rademacher_curve.jpg')

    return None

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def approximate_vc_dim(model):
    return count_parameters(model)

def compute_onet_empirical_risk(model, branch_data, trunk_data, targets):
    with torch.no_grad():
        preds = model(branch_data, trunk_data)
    # targets = targets.reshape(-1, len(trunk_data))
    print(preds.shape, targets.shape)
    return torch.mean((preds-targets)**2).item()

def compute_onet_curves(model, branch_data, trunk_data, targets, max_samples=500, step=10):
    sample_sizes = list(range(step, min(len(branch_data), max_samples) + 1, step))
    # vc_dimensions = []
    empirical_risks = []

    vc_dim = approximate_vc_dim(model)
    # print("VC DIM:", vc_dim)

    for size in sample_sizes:
        branch_subset = branch_data[:size]
        # trunk_subset = trunk_data[:size]
        target_subset = targets[:size]

        risk = compute_onet_empirical_risk(model, branch_subset, trunk_data, target_subset)

        # vc_dimensions.append(vc_dim)
        empirical_risks.append(risk)

    return sample_sizes, vc_dim, empirical_risks

def plot_onet_erm_curves(sample_sizes, vc_dim, erms, erms_base, outdir='.'):
    plt.figure(figsize=(12, 6))
    plt.plot(sample_sizes, erms, label='DeepONet - KAN')
    plt.plot(sample_sizes, erms_base, label='DeepOnet Baseline DenseNet')
    plt.xlabel('Sample Size')
    plt.ylabel('Empirical Risk')
    plt.title('DeepONet Empirical Risk vs Sample Size')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{outdir}/onet_erm_curves.jpg')

    print("KAN Model VC dim: ", vc_dim)
    return None

#Main function.
def main():

    #Command line parser for the model that we want to evaluate.
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('-model', dest='modeltype', type=str, default='efficient_kan',
                            help='Model type.',
                            choices=['densenet', 'efficient_kan', 'original_kan', 'cheby', 'jacobi', 'legendre'])
    model_parser.add_argument('-problem', dest='problem', type=str, default='burgers',
                            help='Problem for analysis.',
                            choices=['burgers', '1d_darcy', '2d_darcy', 'derivative'])
    modeltype = model_parser.parse_args().modeltype
    problem = model_parser.parse_args().problem
    print(f"Running loss metric analysis on modeltype {modeltype}, problem {problem}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if problem == 'burgers':
        #First, load in the data (x,y,z) and the evaluated model.
        #...
        #Data first.
        data = loadmat('data/Burger.mat') # Load the .mat file
        #print(data)
        print(data['tspan'].shape)
        print(data['input'].shape)  # Initial conditions: Gaussian random fields, Nsamples x 101, each IC sample is (1 x 101)
        print(data['output'].shape) # Time evolution of the solution field: Nsamples x 101 x 101.
                                    # Each field is 101 x 101, rows correspond to time and columns respond to location.
                                    # First row corresponds to solution at t=0 (1st time step)
                                    # and next  row corresponds to solution at t=0.01 (2nd time step) and so on.
                                    # last row correspond to solution at t=1 (101th time step).

        # %%
        # Convert NumPy arrays to PyTorch tensors
        inputs = torch.from_numpy(data['input']).float().to(device)
        outputs = torch.from_numpy(data['output']).float().to(device)

        t_span = torch.from_numpy(data['tspan'].flatten()).float().to(device)
        x_span = torch.linspace(0, 1, data['output'].shape[2]).float().to(device)
        nt, nx = len(t_span), len(x_span) # number of discretizations in time and location.
        print("nt =",nt, ", nx =",nx)
        # print("Shape of t-span and x-span:",t_span.shape, x_span.shape)
        # print("t-span:", t_span)
        # print("x-span:", x_span)

        # Estimating grid points
        T, X = torch.meshgrid(t_span, x_span)
        # print(T)
        # print(X)
        grid = torch.vstack((T.flatten(), X.flatten())).T
        print("Shape of grid:", grid.shape) # (nt*nx, 2)
        # print("grid:", grid) # time, location

        #Outputs reshaped as needed for evaluations.
        outputs = outputs.reshape(-1, nt*nx)

        p = 100
        input_neurons_branch = nx
        input_neurons_trunk = 2
        if modeltype == 'densenet':
            # b = DenseNet(layersizes=[input_neurons_branch] + [100]*6 + [p], activation=nn.SiLU())
            # t = DenseNet(layersizes=[input_neurons_trunk] + [100]*6 + [p], activation=nn.SiLU())
            pass
        elif modeltype == 'efficient_kan':
            b = efficient_kan.KAN(layers_hidden=[input_neurons_branch] + [2*input_neurons_branch+1]*1 + [p])
            t = efficient_kan.KAN(layers_hidden=[input_neurons_trunk] + [input_neurons_trunk*2+1]*1 + [p])
            b2 = DenseNet(layersizes=[input_neurons_branch] + [1000] + [p], activation=nn.SiLU())
            t2 = DenseNet(layersizes=[input_neurons_trunk] + [1000] + [p], activation=nn.SiLU())
        elif modeltype == 'original_kan':
            b = kan.KAN(width=[input_neurons_branch,2*input_neurons_branch+1,p], grid=5, k=3, seed=0)
            t = kan.KAN(width=[input_neurons_trunk,2*input_neurons_trunk+1,p], grid=5, k=3, seed=0)
            b2 = DenseNet(layersizes=[input_neurons_branch] + [1000] + [p], activation=nn.SiLU())
            t2 = DenseNet(layersizes=[input_neurons_trunk] + [1000] + [p], activation=nn.SiLU())
        elif modeltype == 'cheby':
            b = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='cheby_kan', layernorm=False)
            t = KANTrunkNet(input_neurons_trunk, 2*input_neurons_trunk+1, p, modeltype='cheby_kan', layernorm=False)
            b2 = DenseNet(layersizes=[input_neurons_branch] + [1000] + [p], activation=nn.SiLU())
            t2 = DenseNet(layersizes=[input_neurons_trunk] + [1000] + [p], activation=nn.SiLU())
        elif modeltype == 'jacobi':
            b = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='jacobi_kan', layernorm=False)
            t = KANTrunkNet(input_neurons_trunk, 2*input_neurons_trunk+1, p, modeltype='jacobi_kan', layernorm=False)
            b2 = DenseNet(layersizes=[input_neurons_branch] + [1000] + [p], activation=nn.SiLU())
            t2 = DenseNet(layersizes=[input_neurons_trunk] + [1000] + [p], activation=nn.SiLU())
        elif modeltype == 'legendre':
            b = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='legendre_kan', layernorm=False)
            t = KANTrunkNet(input_neurons_trunk, 2*input_neurons_trunk+1, p, modeltype='legendre_kan', layernorm=False)
            b2 = DenseNet(layersizes=[input_neurons_branch] + [1000] + [p], activation=nn.SiLU())
            t2 = DenseNet(layersizes=[input_neurons_trunk] + [1000] + [p], activation=nn.SiLU())
        else:
            # TODO: add support for other models here.
            pass
        model = DeepONet(b,t)
        model2 = DeepONet(b2, t2)
        model.load_state_dict(torch.load(f'./DeepONet_results/seed=0/model_state_dict_{modeltype}.pt', weights_only=True))
        model2.load_state_dict(torch.load(f'./DeepONet_results/seed=0/model_state_dict_densenet.pt', weights_only=True))
        model.to(device)
        model2.to(device)
        model.eval()
        model2.eval()

        branch_data = inputs; branch_data.to(device)
        trunk_data = grid; trunk_data.to(device)
        targets = outputs; targets.to(device)

        #Once data and model are loaded, then proceed with the function calls.
        #First up is the loss landscape.
        loss_landscape = compute_loss_landscape_onet(model, branch_data, trunk_data, targets)
        plot_loss_landscape(loss_landscape, f"DeepONet {modeltype} Loss Landscape", f'{modeltype}')
        loss_landscape_baseline = compute_loss_landscape_onet(model2, branch_data, trunk_data, targets)
        plot_loss_landscape(loss_landscape_baseline, f"DeepONet DenseNet Loss Landscape Baseline", 'baseline')

        #Next is the Rademacher compelxity.
        sample_sizes, complexities = compute_rademacher_curve(model, branch_data, trunk_data, device)
        sample_sizes_base, complexities_base = compute_rademacher_curve(model2, branch_data, trunk_data, device)
        # plot_onet_rademacher_curves(sample_sizes, complexities)
        # print(f"Final Rademacher complexity of DeepONet: {rademacher_complexity(model, branch_data, trunk_data, device):.6f}")
        plot_onet_rademacher_curves(sample_sizes, complexities, complexities_base)

        #Finally, empirical risk and VC dimension.
        sample_sizes, vc_dim, risks = compute_onet_curves(model, branch_data, trunk_data, targets)
        sample_sizes_base, vc_dim_base, risks_base = compute_onet_curves(model2, branch_data, trunk_data, targets)
        plot_onet_erm_curves(sample_sizes, vc_dim, risks, risks_base)

        print(f'Loss metrics analysis complete for problem {problem}.')
    elif problem == '1d_darcy':
        train_data_path = 'nonlineardarcy_train.mat'
        test_data_path = 'nonlineardarcy_test.mat'
        train_data_url = 'https://drive.usercontent.google.com/download?id=1DTAkXxMjB2xekkqQt1CLwGeKtz5LP6e1&export=download&authuser=0&confirm=t&uuid=cbe6a789-0751-466b-a95d-ee33ff43d6c1&at=APZUnTVEUevchEMmdzxatZeZ3Saj:1723137744388'
        test_data_url = 'https://drive.usercontent.google.com/download?id=1PsKE5yQkIM8fWBdC51d9VKP55TQZ06OC&export=download&authuser=0&confirm=t&uuid=51001974-28a4-468b-a717-20dc3e80a98e&at=APZUnTVp9b1TpDyRuapd8isoC-WR:1723137779616'  

        data_train = loadmat(train_data_path)
        data_test = loadmat(test_data_path)    

        input_train = torch.from_numpy(data_train['f_train']).to(device).float()
        output_train = torch.from_numpy(data_train['u_train']).to(device).float()
        x_train = torch.from_numpy(data_train['x']).to(device).float().t()

        input_test = torch.from_numpy(data_test['f_test']).to(device).float()
        output_test = torch.from_numpy(data_test['u_test']).to(device).float()
        x_test = torch.from_numpy(data_test['x']).to(device).float().t()

        input_neurons_branch = 50 #This is based on Xingjian's choice, worth changing?
        input_neurons_trunk = 1
        p = 20 #Again, is this worth modifying?

        #Defining the model.
        if modeltype=='efficient_kan':
            branch_net = efficient_kan.KAN(layers_hidden=[input_neurons_branch] + [2*input_neurons_branch+1]*1 + [p])
            trunk_net = efficient_kan.KAN(layers_hidden=[input_neurons_trunk] + [2*input_neurons_trunk+1]*1 + [p])
            branch_net2 = DenseNet(layersizes=[input_neurons_branch] + [10000]*1 + [p], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
            trunk_net2 = DenseNet(layersizes=[input_neurons_trunk] + [10000]*1 + [p], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
        elif modeltype == 'original_kan':
            branch_net = kan.KAN(width=[input_neurons_branch,2*input_neurons_branch+1,p], grid=5, k=3, seed=0)
            trunk_net = kan.KAN(width=[input_neurons_trunk,2*input_neurons_trunk+1,p], grid=5, k=3, seed=0)
            branch_net2 = DenseNet(layersizes=[input_neurons_branch] + [10000]*1 + [p], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
            trunk_net2 = DenseNet(layersizes=[input_neurons_trunk] + [10000]*1 + [p], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
        elif modeltype == 'cheby':
            branch_net = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='cheby_kan', layernorm=False)
            trunk_net = KANTrunkNet(input_neurons_trunk, 2*input_neurons_trunk+1, p, modeltype='cheby_kan', layernorm=False)
            branch_net2 = DenseNet(layersizes=[input_neurons_branch] + [10000]*1 + [p], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
            trunk_net2 = DenseNet(layersizes=[input_neurons_trunk] + [10000]*1 + [p], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
        elif modeltype == 'jacobi':
            branch_net = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='jacobi_kan', layernorm=False)
            trunk_net = KANTrunkNet(input_neurons_trunk, 2*input_neurons_trunk+1, p, modeltype='jacobi_kan', layernorm=False)
            branch_net2 = DenseNet(layersizes=[input_neurons_branch] + [10000]*1 + [p], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
            trunk_net2 = DenseNet(layersizes=[input_neurons_trunk] + [10000]*1 + [p], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
        elif modeltype == 'legendre':
            branch_net = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='legendre_kan', layernorm=False)
            trunk_net = KANTrunkNet(input_neurons_trunk, 2*input_neurons_trunk+1, p, modeltype='legendre_kan', layernorm=False)
            branch_net2 = DenseNet(layersizes=[input_neurons_branch] + [10000]*1 + [p], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
            trunk_net2 = DenseNet(layersizes=[input_neurons_trunk] + [10000]*1 + [p], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
        else:
            pass
        model = DeepONet(branch_net, trunk_net)
        model2 = DeepONet(branch_net2, trunk_net2)
        model.load_state_dict(torch.load(f'./1D_Darcy_DeepONet/deeponet_model_{modeltype}.pt', weights_only=True))
        model2.load_state_dict(torch.load(f'./1D_Darcy_DeepONet/deeponet_model_densenet.pt', weights_only=True))
        model.to(device)
        model2.to(device)
        model.eval()
        model2.eval()

        branch_data = input_train; branch_data.to(device)
        trunk_data = x_train; trunk_data.to(device)
        targets = output_train; targets.to(device)

        #Once data and model are loaded, then proceed with the function calls.
        #First up is the loss landscape.
        loss_landscape = compute_loss_landscape_onet(model, branch_data, trunk_data, targets)
        plot_loss_landscape(loss_landscape, f"DeepONet {modeltype} {problem} Loss Landscape", f'{modeltype}', outdir='./1D_Darcy_DeepONet')
        loss_landscape_baseline = compute_loss_landscape_onet(model2, branch_data, trunk_data, targets)
        plot_loss_landscape(loss_landscape_baseline, f"DeepONet DenseNet {problem} Loss Landscape Baseline", 'baseline', outdir='./1D_Darcy_DeepONet')

        #Next is the Rademacher compelxity.
        sample_sizes, complexities = compute_rademacher_curve(model, branch_data, trunk_data, device, max_samples=1000)
        sample_sizes_base, complexities_base = compute_rademacher_curve(model2, branch_data, trunk_data, device, max_samples=1000)
        # plot_onet_rademacher_curves(sample_sizes, complexities)
        # print(f"Final Rademacher complexity of DeepONet: {rademacher_complexity(model, branch_data, trunk_data, device):.6f}")
        plot_onet_rademacher_curves(sample_sizes, complexities, complexities_base, outdir='./1D_Darcy_DeepONet')

        #Finally, empirical risk and VC dimension.
        sample_sizes, vc_dim, risks = compute_onet_curves(model, branch_data, trunk_data, targets, max_samples=1000)
        sample_sizes_base, vc_dim_base, risks_base = compute_onet_curves(model2, branch_data, trunk_data, targets, max_samples=1000)
        plot_onet_erm_curves(sample_sizes, vc_dim, risks, risks_base, outdir='./1D_Darcy_DeepONet')

        print(f'Loss metrics analysis complete for problem {problem}.')
    elif problem == '2d_darcy':
        pass
        # # load and process data
        # train_data_path = 'linearDarcy_train.mat'
        # test_data_path = 'linearDarcy_test.mat'
        # train_data_url = 'https://drive.usercontent.google.com/download?id=1dSjOjI-DHhmUgcFonAqLIqvQIfKouqZo&export=download&authuser=0&confirm=t&uuid=90fa75a0-f578-4d6c-856e-99081265b1d3&at=APZUnTVQ7tBgzAs96tmlIzyl9Z9u:1723137673435'  # Replace with actual URL
        # test_data_url = 'https://drive.usercontent.google.com/download?id=1_6s-fPzTZCqpysLhfocm6qth8OBl1H-k&export=download&authuser=0&confirm=t&uuid=1a727246-2e09-4f82-a65c-fb2234f105b1&at=APZUnTVNqcWgyHb2nmhkk0jydRL9:1723137701171'   # Replace with actual URL
        
        # data_train = loadmat(train_data_path)
        # data_test = loadmat(test_data_path)

        # f1_train = torch.from_numpy(data_train['f1_train']).to(device).float()   
        # f1_train = f1_train.reshape(f1_train.shape[0], -1)      # 51000 by 961
        # f2_train = torch.from_numpy(data_train['f2_train']).to(device).float()
        # f2_train = f2_train.reshape(f2_train.shape[0], -1)
        # u_train = torch.from_numpy(data_train['u_train']).to(device).float()   # 51000 by 450
        # x = torch.from_numpy(data_train['x']).to(device).float()
        # y = torch.from_numpy(data_train['y']).to(device).float()
        # x_train = torch.cat((x, y), dim=1)     # 450 by 2

        # f1_test = torch.from_numpy(data_test['f1_test']).to(device).float()
        # f1_test = f1_test.reshape(f1_test.shape[0], -1)
        # f2_test = torch.from_numpy(data_test['f2_test']).to(device).float()
        # f2_test = f2_test.reshape(f2_test.shape[0], -1)
        # u_test = torch.from_numpy(data_test['u_test']).to(device).float()
        # x = torch.from_numpy(data_test['x']).to(device).float()
        # y = torch.from_numpy(data_test['y']).to(device).float()
        # x_test = torch.cat((x, y), dim=1)

        # # define and set up model
        # sensor_size = f1_train.shape[1]
        # input_neurons_branch = sensor_size
        # input_neurons_trunk = 2
        # p = 64

        # if modeltype=='efficient_kan':
        #     branch_net1 = efficient_kan.KAN(layers_hidden=[input_neurons_branch]+[2*input_neurons_branch]*1+[p])
        #     branch_net2 = efficient_kan.KAN(layers_hidden=[input_neurons_branch]+[2*input_neurons_branch]*1+[p])
        #     branch_net1.to(device)
        #     branch_net2.to(device)
        #     trunk_net = efficient_kan.KAN(layers_hidden=[input_neurons_trunk]+[2*input_neurons_trunk]*1+[p])
        #     trunk_net.to(device)
        # elif modeltype == 'original_kan':
        #     branch_net1 = kan.KAN(width=[input_neurons_branch, 2*input_neurons_branch+1, p], grid=5, k=3, seed=0)
        #     branch_net2 = kan.KAN(width=[input_neurons_branch, 2*input_neurons_branch+1, p], grid=5, k=3, seed=0)
        #     branch_net1.to(device)
        #     branch_net2.to(device)
        #     trunk_net = kan.KAN(width=[input_neurons_trunk, 2*input_neurons_trunk+1, p], grid=5, k=3, seed=0)
        #     trunk_net.to(device)
        # elif modeltype == 'cheby':
        #     branch_net1 = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='cheby_kan', layernorm=False)
        #     branch_net2 = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='cheby_kan', layernorm=False)
        #     branch_net1.to(device)
        #     branch_net2.to(device)
        #     trunk_net = KANBranchNet(input_neurons_trunk, 2*input_neurons_trunk+1, p, modeltype='cheby_kan', layernorm=False)
        #     trunk_net.to(device)
        # elif modeltype == 'jacobi':
        #     branch_net1 = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='jacobi_kan', layernorm=False)
        #     branch_net2 = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='jacobi_kan', layernorm=False)
        #     branch_net1.to(device)
        #     branch_net2.to(device)
        #     trunk_net = KANBranchNet(input_neurons_trunk, 2*input_neurons_trunk+1, p, modeltype='jacobi_kan', layernorm=False)
        #     trunk_net.to(device)
        # elif modeltype == 'legendre':
        #     branch_net1 = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='legendre_kan', layernorm=False)
        #     branch_net2 = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='legendre_kan', layernorm=False)
        #     branch_net1.to(device)
        #     branch_net2.to(device)
        #     trunk_net = KANBranchNet(input_neurons_trunk, 2*input_neurons_trunk+1, p, modeltype='legendre_kan', layernorm=False)
        #     trunk_net.to(device)
        # else:
        #     branch_net1 = DenseNet(layersizes=[sensor_size] + [1000]*1 + [p], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
        #     branch_net2 = DenseNet(layersizes=[sensor_size] + [1000]*1 + [p], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
        #     branch_net1.to(device)
        #     branch_net2.to(device)
        #     trunk_net = DenseNet(layersizes=[input_neurons_trunk] + [1000]*1 + [p], activation=nn.ReLU()) #nn.LeakyReLU() #nn.Tanh()
        #     trunk_net.to(device)
        # model = DeepONet(branch_net1, branch_net2, trunk_net)
        # model.to(device)
    else:
        print("Invalid problem passed for analysis.")
        return

if __name__  == '__main__':
    main()

