import numpy as np
import torch
from torch import nn
import random
import os
from scipy.io import loadmat
import sys
import argparse

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import urllib.request
from scipy.interpolate import griddata
from darcy_2d_deeponet import create_model as create_model_2d_darcy
from darcy_2d_deeponet import load_data as load_data_2d_darcy
from darcy_2d_deeponet import plot_results as plot_results_2d_darcy
from darcy_1d_deeponet import create_model as create_model_1d_darcy
from darcy_1d_deeponet import load_data as load_data_1d_darcy
from darcy_1d_deeponet import plot_results as plot_results_1d_darcy
# from DeepONet import create_model as create_model_burgers
# from DeepONet import load_data as load_data_burgers
# from DeepONet import plot_results as plot_results_burgers

def load_and_plot():
    pass

def main():
    parser = argparse.ArgumentParser(description='Plotting from saved model.')
    parser.add_argument('-problem', dest='problem', type=str, help='Problem for which we want to plot.')
    parser.add_argument('-modeltype', dest='modeltype', type=str, help='Architecture we want to plot for.',
                        choices=['densenet', 'efficient_kan', 'original_kan', 'cheby', 'jacobi', 'legendre'])
    parser.add_argument('-mode', dest='mode', type=str, help='Model mode we are plotting.',
                        choices=['deep', 'shallow'])
    args = parser.parse_args()
    problem = args.problem
    modeltype = args.modeltype
    mode = args.mode

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #NOTE: all saved model .pt files are saved as state dicts i.e. torch.save(model.state_dict()).
    #Thus in order to load them correctly, it is necessary to setup an empty model with correct sizing,
    #and then load the state dict into that empty model.

    if problem == '2d_darcy':
        model_path = f'./2D_Darcy_DeepONet/{mode}/{modeltype}_deeponet_model.pt'
        model = create_model_2d_darcy(modeltype, mode, device)
        output_dir = f'./2D_Darcy_DeepONet/{mode}'

        model.load_state_dict(torch.load(model_path))
        f1_train, f2_train, x_train, u_train, f1_test, f2_test, x_test, u_test = load_data_2d_darcy(device)
        plot_results_2d_darcy(f1_test, f2_test, x_test, u_test, model, modeltype, output_dir)
        print("2D Darcy plots complete.")
    elif problem == '1d_darcy':
        model_path = f'./1D_Darcy_DeepONet/{mode}/deeponet_model_{modeltype}.pt'
        model = create_model_1d_darcy(modeltype, mode, device)
        output_dir = f'./1D_Darcy_DeepONet/{mode}'

        model.load_state_dict(torch.load(model_path))
        input_train, output_train, x_train, input_test, output_test, x_test = load_data_1d_darcy(device)
        preds = model(input_test, x_test)
        plot_results_1d_darcy(preds, output_test, x_test, output_dir, modeltype)
        print("1D Darcy plots complete.")
    else:
        model_path = f'./DeepONet_results/{mode}/model_state_dict_{modeltype.pt}'
        model = create_model_burgers(modeltype, mode, device)
        output_dir = f'./DeepONet_results/{mode}'

        model.load_state_dict(torch.load(model_path))
        

    return None

if __name__ == '__main__':
    main()


