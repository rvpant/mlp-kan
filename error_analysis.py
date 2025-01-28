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
from darcy_2d_deeponet import test_error_analysis as test_error_analysis_2d_darcy
from darcy_1d_deeponet import create_model as create_model_1d_darcy
from darcy_1d_deeponet import load_data as load_data_1d_darcy
from darcy_1d_deeponet import plot_results as plot_results_1d_darcy
from darcy_1d_deeponet import test_error_analysis as test_error_analysis_1d_darcy
from DeepONet import create_model as create_model_burgers
from DeepONet import load_data as load_data_burgers
# from DeepONet import plot_results as plot_results_burgers
from DeepONet import test_error_analysis as test_error_analysis_burgers

def noise_analysis(noise, model, problem, device):
    if problem == '1d_darcy':
        testdata = loadmat(f'nonlineardarcy_test_noise_{noise}.mat')
        test_input, test_output, test_x = torch.from_numpy(testdata['f_test']).to(device).float(), torch.from_numpy(testdata['u_test']).to(device).float(), torch.from_numpy(testdata['x']).to(device).float().t()

        noisy_preds = model(test_input, test_x).detach().cpu().numpy()
        test_output = test_output.detach().cpu().numpy()
        # print("Test predictions shape: ", np.shape(noisy_preds))
        # print("Test output shape: ", np.shape(test_output))
        abs_errors = np.abs(noisy_preds - test_output)
        l2_errors = []
        for pred, true in zip(noisy_preds, test_output):
            # print(np.shape(pred), np.shape(true))
            num = np.linalg.norm(pred-true, ord=2)
            denom = np.linalg.norm(true)
            l2_errors.append(num/denom)
        return abs_errors, l2_errors
    elif problem == 'burgers':
        pass
    return

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
        print("Starting 2D Darcy error analysis.")
        model_path = f'./2D_Darcy_DeepONet/{mode}/{modeltype}_deeponet_model.pt'; print(f"Model Path : {model_path}")
        model = create_model_2d_darcy(modeltype, mode, device)
        output_dir = f'./2D_Darcy_DeepONet/{mode}'

        model.load_state_dict(torch.load(model_path))
        f1_train, f2_train, x_train, u_train, f1_test, f2_test, x_test, u_test = load_data_2d_darcy(device)
        abserr, l2err = test_error_analysis_2d_darcy(f1_test, f2_test, x_test, u_test, model, modeltype, output_dir)
        # print(f"Absolute errors mean: {np.mean(abserr)}")
        print(f"Relative L2 error mean: {np.mean(l2err)}")
        print(f"Relative L2 error stdev: {np.std(l2err)}")
        print(f"Worst-case relative L2 error: {np.max(l2err)}")
        print("L2 errors shape: ", np.shape(l2err))
        print("2D Darcy error analysis complete.")
    elif problem == '1d_darcy':
        print("Starting 1D Darcy error analysis.")
        model_path = f'./1D_Darcy_DeepONet/{mode}/deeponet_model_{modeltype}.pt'
        model = create_model_1d_darcy(modeltype, mode, device)
        output_dir = f'./1D_Darcy_DeepONet/{mode}'

        model.load_state_dict(torch.load(model_path))
        input_train, output_train, x_train, input_test, output_test, x_test = load_data_1d_darcy(device)
        preds = model(input_test, x_test)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameter count is {total_params}")
        abserr, l2err = test_error_analysis_1d_darcy(preds, output_test, x_test, output_dir, modeltype)
        # print(f"Absolute errors mean: {np.mean(np.sum(abserr, axis=1))}")
        print(f"Relative L2 error mean: {np.mean(l2err)}")
        print(f"Relative L2 error stdev: {np.std(l2err)}")
        print(f"Worst-case relative L2 error: {np.max(l2err)}")
        # print("L2 errors: ", l2err)
        print("1D Darcy error analysis complete.")

        #ADD function call for noisy analysis.
        for n in [0.01, 0.05, 0.1]:
            print(f"Noise = {n}")
            noisy_abserr, noisy_l2err = noise_analysis(n, model, problem, device)
            # print(f"Noisy absolute errors mean: {np.mean(np.sum(noisy_abserr, axis=1))}")
            print(f"Noisy Relative L2 error mean: {np.mean(noisy_l2err)}")
            print(f"Noisy Relative L2 error std: {np.std(noisy_l2err)}")
            print(f"Noisy Worst-case relative L2 error: {np.max(noisy_l2err)}")
            # print("Noisy L2 errors: ", noisy_l2err)

    else:
        model_path = f'./DeepONet_results/{mode}/seed=0/model_state_dict_{modeltype}.pt'
        model = create_model_burgers(modeltype, mode, device)
        output_dir = f'./DeepONet_results/{mode}'

        model.load_state_dict(torch.load(model_path))
        inputs_train, inputs_test, outputs_train, outputs_test, grid, nx = load_data_burgers(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameter count is {total_params}")

        mses, l2err = test_error_analysis_burgers(inputs_test, outputs_test, grid, nx, model, modeltype, output_dir)
        print(f"Relative L2 error mean: {np.mean(l2err)}")
        print(f"Relative L2 error stdev: {np.std(l2err)}")
        print(f"Worst-case relative L2 error: {np.max(l2err)}")
        print(f"MSE mean: {np.mean(mses)}")
        print(f"MSE stdev: {np.std(mses)}")
        print(f"Worst-case MSE: {np.max(mses)}")
        print("Burgers error analysis complete.")

    return None


if __name__ == '__main__':
    main()
