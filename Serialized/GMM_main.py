# -*- coding: utf-8 -*-

from __future__ import division

import copy
from DISCO import DISCO
from DINGO import DINGO
from GIANT import GIANT
from GMM_data_loader import GMM_data
from inexactDANE import inexactDANE
import matplotlib.pyplot as plt
import numpy as np
from SGD import SGD
import torch
from worker import Worker

# The following 3 lines prevent matplotlib from printing the mplDeprecation warning during runtime
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


def main():
    use_cuda = True # use CUDA/GPU? If False then CPU is used
    num_runs = 100
    num_train_samples = 20000
    num_train_features = 100
    objective_function_lambda_max = 1.9
    gradient_lambda_max = 8
    test_lambda_max = 60

    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)

    # uncomment which algorithms to run
    algorithms = ['DINGO',
                  'GIANT',
                  'DISCO',
                  'InexactDANE',
                  'Asynchronous SGD',
                  'Synchronous SGD'
                 ]

    print("Algorithms: " + ', '.join(algorithms))
    print("Whole Training Dataset Size: " + str((num_train_samples, num_train_features)))

    num_partitions = 4

    config = {'num_partitions' : num_partitions,
              'max_iterations' : 10000,
              'max_communication_rounds' : 20,
              'grad_tol' : 1e-8, # if norm(gradient)<grad_tol then algorithm loop breaks
              'lamda' : 0, # regularization parameter
              'subproblem_max_iterations' : 50, # does not apply to Async-SGD and Sync-SGD
              'subproblem_tolerance' : 1e-4, # does not apply to Async-SGD and Sync-SGD
              'use_preconditioning_in_subproblem' : False, # does not apply to Async-SGD and Sync-SGD
              'line_search_max_iterations' : 100, # used by GIANT and DINGO
              'line_search_rho' : 1e-4, # used by GIANT and DINGO
              'line_search_start_val' : 1e+0, # used by GIANT and DINGO
              'DINGO_theta' : 1e-4, # used by DINGO
              'DINGO_phi' : 1e-6, # used by DINGO
              'DISCO_mu' : 1, # used by DiSCO when preconditioning is used
              'inexactDANE_SVRG_stepsize' : 1e-1, # used by InexactDANE and AIDE
              'inexactDANE_eta' : 1, # used by InexactDANE and AIDE
              'inexactDANE_mu' : 0, # used by InexactDANE and AIDE
              'AIDE_tau' : 1e+0, # used by AIDE
              'Asynchronous_SGD_stepsize' : 1e-2, # used by Async-SGD
              'Synchronous_SGD_stepsize' : 1e-1, # used by Sync-SGD
              'SGD_minibatch_size' : (num_train_samples//num_partitions)//5, # used by Async-SGD and Sync-SGD
              'obj_fun' : 'GMM',
              'dataset' : 'Random',
              'algorithms' : algorithms,
              'use_multiprocessing' : False,
              'load_plot_data' : False,
              'save_plot_data' : False,
              'dataset_train_X_size' : str((num_train_samples, num_train_features)),
              'dataset_train_Y_size' : str((num_train_samples, 1))}

    d = 2*num_train_features+1 # number of weights

    # uncomment which starting point to use
    starting_point = [#'randn',
                      #'ones',
                      'zeros']
    config['w0_str'] = starting_point[0]

    if config['w0_str'] == 'zeros':
        w0 = torch.zeros((d,1)) # [d,1]
    elif config['w0_str'] == 'ones':
        w0 = torch.ones((d,1))
    else:
        w0 = 1*torch.randn(d,1)

    config['w0'] = w0

    train_Y = torch.zeros(1,1) # this is unused in the performance profiles
    test_Y = torch.zeros(1,1) # this is unused in the performance profiles

    relative_final_loss_dictionary = {}
    relative_final_grad_norm_dictionary = {}
    relative_final_estimation_error_dictionary = {}

    for algorithm in algorithms:
        relative_final_loss_dictionary[algorithm] = []
        relative_final_grad_norm_dictionary[algorithm] = []
        relative_final_estimation_error_dictionary[algorithm] = []

    for k in range(num_runs):
        train_X, test_X, C1, C2, c1, c2, true_weights = GMM_data(num_train_samples, num_train_features, 1, k)
        config['GMM_C1'] = C1.type(torch.Tensor)
        config['GMM_C2'] = C2.type(torch.Tensor)
        config['GMM_C1_determinant'] = 1/c1.type(torch.Tensor)
        config['GMM_C2_determinant'] = 1/c2.type(torch.Tensor)
        true_weights = true_weights.type(torch.Tensor)

        workers = []
        train_X = train_X.type(torch.Tensor)
        test_X = test_X.type(torch.Tensor)
        workers_config = copy.deepcopy(config)

        for k in range(config['num_partitions']):
            s = num_train_samples // config['num_partitions']
            x, y = train_X[k*s:(k+1)*s], train_Y[k*s:(k+1)*s]
            workers.append(Worker(x,y,workers_config))

        for algorithm in algorithms:

            if algorithm == 'GIANT':
                final_weights_GIANT, final_loss_GIANT, final_grad_norm_GIANT, final_test_accuracy_GIANT = GIANT(workers, copy.deepcopy(config), test_X, test_Y)
                estimation_error_GIANT = 0.5*torch.abs(final_weights_GIANT[0:1]-true_weights[0:1])/torch.abs(true_weights[0:1]) + 0.5*torch.norm(final_weights_GIANT[1:]-true_weights[1:])/torch.norm(true_weights[1:])

            if algorithm == 'DINGO':
                final_weights_DI, final_loss_DI, final_grad_norm_DI, final_test_accuracy_DI = DINGO(workers, copy.deepcopy(config), test_X, test_Y)
                estimation_error_DI = 0.5*torch.abs(final_weights_DI[0:1]-true_weights[0:1])/torch.abs(true_weights[0:1]) + 0.5*torch.norm(final_weights_DI[1:]-true_weights[1:])/torch.norm(true_weights[1:])

            if algorithm == 'DISCO':
                final_weights_DISCO, final_loss_DISCO, final_grad_norm_DISCO, final_test_accuracy_DISCO = DISCO(workers, copy.deepcopy(config), test_X, test_Y)
                estimation_error_DISCO = 0.5*torch.abs(final_weights_DISCO[0:1]-true_weights[0:1])/torch.abs(true_weights[0:1]) + 0.5*torch.norm(final_weights_DISCO[1:]-true_weights[1:])/torch.norm(true_weights[1:])

            if algorithm == 'InexactDANE':
                final_weights_InexactDANE, final_loss_InexactDANE, final_grad_norm_InexactDANE, final_test_accuracy_InexactDANE = inexactDANE(workers, copy.deepcopy(config), test_X, test_Y)
                estimation_error_InexactDANE = 0.5*torch.abs(final_weights_InexactDANE[0:1]-true_weights[0:1])/torch.abs(true_weights[0:1]) + 0.5*torch.norm(final_weights_InexactDANE[1:]-true_weights[1:])/torch.norm(true_weights[1:])

            if algorithm == 'Asynchronous SGD':
                final_weights_ASGD, final_loss_ASGD, final_grad_norm_ASGD, final_test_accuracy_ASGD = SGD(workers, copy.deepcopy(config), 'asynchronous', test_X, test_Y)
                estimation_error_ASGD = 0.5*torch.abs(final_weights_ASGD[0:1]-true_weights[0:1])/torch.abs(true_weights[0:1]) + 0.5*torch.norm(final_weights_ASGD[1:]-true_weights[1:])/torch.norm(true_weights[1:])

            if algorithm == 'Synchronous SGD':
                final_weights_SSGD, final_loss_SSGD, final_grad_norm_SSGD, final_test_accuracy_SSGD = SGD(workers, copy.deepcopy(config), 'synchronous', test_X, test_Y)
                estimation_error_SSGD = 0.5*torch.abs(final_weights_SSGD[0:1]-true_weights[0:1])/torch.abs(true_weights[0:1]) + 0.5*torch.norm(final_weights_SSGD[1:]-true_weights[1:])/torch.norm(true_weights[1:])

        best_final_loss = min(final_loss_GIANT, final_loss_DI, final_loss_DISCO, final_loss_InexactDANE, final_loss_ASGD, final_loss_SSGD)
        assert(best_final_loss > 0)
        relative_final_loss_dictionary['GIANT'].append(final_loss_GIANT/best_final_loss)
        relative_final_loss_dictionary['DINGO'].append(final_loss_DI/best_final_loss)
        relative_final_loss_dictionary['DISCO'].append(final_loss_DISCO/best_final_loss)
        relative_final_loss_dictionary['InexactDANE'].append(final_loss_InexactDANE/best_final_loss)
        relative_final_loss_dictionary['Asynchronous SGD'].append(final_loss_ASGD/best_final_loss)
        relative_final_loss_dictionary['Synchronous SGD'].append(final_loss_SSGD/best_final_loss)

        best_final_grad_norm = min(final_grad_norm_GIANT, final_grad_norm_DI, final_grad_norm_DISCO, final_grad_norm_InexactDANE, final_grad_norm_ASGD, final_grad_norm_SSGD)
        relative_final_grad_norm_dictionary['GIANT'].append(final_grad_norm_GIANT/best_final_grad_norm)
        relative_final_grad_norm_dictionary['DINGO'].append(final_grad_norm_DI/best_final_grad_norm)
        relative_final_grad_norm_dictionary['DISCO'].append(final_grad_norm_DISCO/best_final_grad_norm)
        relative_final_grad_norm_dictionary['InexactDANE'].append(final_grad_norm_InexactDANE/best_final_grad_norm)
        relative_final_grad_norm_dictionary['Asynchronous SGD'].append(final_grad_norm_ASGD/best_final_grad_norm)
        relative_final_grad_norm_dictionary['Synchronous SGD'].append(final_grad_norm_SSGD/best_final_grad_norm)

        best_estimation_error = min(estimation_error_GIANT, estimation_error_DI, estimation_error_DISCO, estimation_error_InexactDANE, estimation_error_ASGD, estimation_error_SSGD)
        relative_final_estimation_error_dictionary['GIANT'].append(estimation_error_GIANT/best_estimation_error)
        relative_final_estimation_error_dictionary['DINGO'].append(estimation_error_DI/best_estimation_error)
        relative_final_estimation_error_dictionary['DISCO'].append(estimation_error_DISCO/best_estimation_error)
        relative_final_estimation_error_dictionary['InexactDANE'].append(estimation_error_InexactDANE/best_estimation_error)
        relative_final_estimation_error_dictionary['Asynchronous SGD'].append(estimation_error_ASGD/best_estimation_error)
        relative_final_estimation_error_dictionary['Synchronous SGD'].append(estimation_error_SSGD/best_estimation_error)

    objective_function_axis = np.linspace(1,objective_function_lambda_max,1000)
    gradient_axis = np.logspace(1,gradient_lambda_max,1000)
    test_axis = np.linspace(1,test_lambda_max,1000)
    plot_final_loss_dictionary = {}
    plot_final_grad_norm_dictionary = {}
    plot_final_estimation_error_dictionary = {}

    for algorithm in algorithms:
        plot_final_loss_dictionary[algorithm] = []
        plot_final_grad_norm_dictionary[algorithm] = []
        plot_final_estimation_error_dictionary[algorithm] = []

    for algorithm in algorithms:
        for lamda in objective_function_axis:
            plot_final_loss_dictionary[algorithm].append(sum(1 for rel_loss in relative_final_loss_dictionary[algorithm] if rel_loss <= lamda)/num_runs)
        for lamda in gradient_axis:
            plot_final_grad_norm_dictionary[algorithm].append(sum(1 for rel_grad_norm in relative_final_grad_norm_dictionary[algorithm] if rel_grad_norm <= lamda)/num_runs)
        for lamda in test_axis:
            plot_final_estimation_error_dictionary[algorithm].append(sum(1 for rel_estimation_error in relative_final_estimation_error_dictionary[algorithm] if rel_estimation_error <= lamda)/num_runs)

    for algorithm in algorithms:
        label = algorithm
        marker = None
        if algorithm == 'GIANT':
            colour = 'g'
            style = '-.'
        if algorithm == 'DINGO':
            colour = 'k'
            style = '-'
        if algorithm == 'DISCO':
            colour = 'c'
            style = '-.'
        if algorithm == 'InexactDANE':
            colour = 'm'
            style = ':'
        if algorithm == 'Asynchronous SGD':
            colour = 'y'
            style = '--'
        if algorithm == 'Synchronous SGD':
            colour = 'b'
            style = '--'

        plt.figure(10, figsize=(4,12))
        plt.subplots_adjust(hspace = 0.4)

        plt.subplot(311)
        plt.plot(objective_function_axis, plot_final_loss_dictionary[algorithm], color=colour, linestyle=style, label=label)
        plt.xlabel(r'$\lambda$')
        plt.ylabel('Performance profile')
        plt.title(r'Objective Function: $f\:(\mathbf{w})$')

        plt.subplot(312)
        plt.semilogx(gradient_axis, plot_final_grad_norm_dictionary[algorithm], color=colour, linestyle=style, label=label, marker=marker)
        plt.xlabel(r'$\lambda$')
        plt.ylabel('Performance profile')
        plt.title(r'Gradient Norm: $||\nabla f(\mathbf{w})||$')

        plt.subplot(313)
        plt.plot(test_axis, plot_final_estimation_error_dictionary[algorithm], color=colour, linestyle=style, label=label)
        plt.xlabel(r'$\lambda$')
        plt.ylabel('Performance profile')
        plt.title('Estimation Error')

    plt.savefig('./Plots/' + 'GMM' + '_' + str(config['num_partitions']) + '.pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
