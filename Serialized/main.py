# -*- coding: utf-8 -*-

from __future__ import division

from AIDE import AIDE
import copy
from DISCO import DISCO
from DINGO import DINGO
from GIANT import GIANT
from inexactDANE import inexactDANE
from myLoadData import myDataLoader
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
    use_multiprocessing = False # parallelize subproblem solvers?
    load_plot_data = True
    save_plot_data = True

    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)

    seed = 1
    torch.manual_seed(seed)

    # uncomment which algorithms to run
    algorithms = ['DINGO',
                  'GIANT',
                  'DISCO',
                  'InexactDANE',
                  'AIDE',
                  'Asynchronous SGD',
                  'Synchronous SGD'
                 ]

    # select objective function to use
    objective_function = [#'Autoencoder',
                          'softmax'
                          ]
    objective_function = objective_function[0]

    # select dataset to use
    data = [#'EMNIST_digits',
            'CIFAR10',
            #'Curves',
            ]
    data = data[0]

    # select starting point to use
    starting_point = [#'random',
                      #'ones',
                      'zeros'
                      ]
    starting_point = starting_point[0]

    data_dir = '../Data/' + data
    DL = myDataLoader(data, data_dir) # the myDataLoader class loads and processes data
    train_X, train_Y, idx = DL.load_train_data()
    test_X, test_Y, _ = DL.load_test_data()

    print("Algorithms: " + ', '.join(algorithms))
    print("Dataset: " + data)
    print("Whole Training Dataset Size: " + str(train_X.shape))

    n, p = train_X.shape

    autoencoder_layers = None
    if objective_function == 'Autoencoder':
        autoencoder_layers = [p,400,300,200,100,50,25,12,6,12,25,50,100,200,300,400,p]
        print("Autoencoder Layers: " + str(autoencoder_layers))

    # we process data for the objective function
    train_X, train_Y, d = DL.process_data(train_X, train_Y, idx, objective_function, autoencoder_layers)
    test_X, test_Y, _ = DL.process_data(test_X, test_Y, idx, objective_function, autoencoder_layers, is_test_data=True)
    print("Test Dataset Size: " + str(test_X.shape))
    test_X = test_X.type(torch.Tensor)
    test_Y = test_Y.type(torch.Tensor)

    num_partitions = 10 # number of workers
    assert(n%num_partitions == 0)

    config = {'num_partitions' : num_partitions,
              'max_iterations' : 10000,
              'max_communication_rounds' : 500,
              'grad_tol' : 1e-8, # if norm(gradient)<grad_tol then algorithm loop breaks
              'lamda' : 1/n, # regularization parameter
              'subproblem_max_iterations' : 50, # does not apply to Async-SGD and Sync-SGD
              'subproblem_tolerance' : 1e-4, # does not apply to Async-SGD and Sync-SGD
              'use_preconditioning_in_subproblem' : False, # does not apply to Async-SGD and Sync-SGD
              'line_search_max_iterations' : 100, # used by GIANT and DINGO
              'line_search_rho' : 1e-4, # used by GIANT and DINGO
              'line_search_start_val' : 1e+0, # used by GIANT and DINGO
              'DINGO_theta' : 1e-4, # used by DINGO
              'DINGO_phi' : 1e-6, # used by DINGO
              'DISCO_mu' : 1, # used by DiSCO when preconditioning is used
              'inexactDANE_SVRG_stepsize' : 1e-3, # used by InexactDANE and AIDE
              'inexactDANE_eta' : 1, # used by InexactDANE and AIDE
              'inexactDANE_mu' : 0, # used by InexactDANE and AIDE
              'AIDE_tau' : 1e+2, # used by AIDE
              'Asynchronous_SGD_stepsize' : 1e-3, # used by Async-SGD
              'Synchronous_SGD_stepsize' : 1e-2, # used by Sync-SGD
              'SGD_minibatch_size' : (n//num_partitions)//5, # used by Async-SGD and Sync-SGD
              'obj_fun' : objective_function,
              'dataset' : data,
              'algorithms' : algorithms,
              'use_multiprocessing' : use_multiprocessing,
              'load_plot_data' : load_plot_data,
              'save_plot_data' : save_plot_data,
              'dataset_train_X_size' : str(train_X.shape),
              'dataset_train_Y_size' : str(train_Y.shape),
              'w0_str' : starting_point}

    if starting_point == 'zeros':
        w0 = torch.zeros((d,1))
    elif starting_point == 'ones':
        w0 = torch.ones((d,1))
    else:
        if objective_function == 'Autoencoder': # default random initialization of PyTorch
            ws = []
            for k in range(len(autoencoder_layers)-1):
                M = torch.zeros((autoencoder_layers[k+1]*autoencoder_layers[k],1))
                b = torch.zeros((autoencoder_layers[k+1],1))
                bound = 1.0/np.sqrt(autoencoder_layers[k])
                ws.append(M.uniform_(-bound,bound))
                ws.append(b.uniform_(-bound,bound))
            w0 = torch.cat(ws, dim=0)
        else:
            w0 = torch.randn(d,1)

    config['w0'] = w0

    if objective_function == 'Autoencoder':
        config['autoencoder_layers'] = autoencoder_layers

    if objective_function == 'GMM': # initialize covariance matrices for GMM
        A1 = torch.randn(p,p)
        A2 = torch.rand(p,p)
        Q1 = torch.qr(A1)[0]
        Q2 = torch.qr(A2)[0]
        D1 = torch.diag(torch.logspace(0,1,steps=p))
        D2 = torch.diag(torch.logspace(1,0,steps=p))
        config['GMM_C1'] = torch.mm(D1,Q1)
        config['GMM_C2'] = torch.mm(D2,Q2)
        config['GMM_C1_determinant'] = torch.det(config['GMM_C1'])
        config['GMM_C2_determinant'] = torch.det(config['GMM_C2'])

    print("Number of Workers: " + str(config['num_partitions']))
    print("Workers' Local Sample Size: " + str(n//config['num_partitions']))
    print("Objective Function: " + objective_function)
    print("Problem Dimension: " + str(d))
    print("Regularization Parameter: " + str(config['lamda']))
    print("Maximum Iterations: " + str(config['max_iterations']))
    print("Maximum Communication Rounds: " + str(config['max_communication_rounds']))

    # Initialize workers
    workers = []
    train_X = train_X.type(torch.Tensor)
    train_Y = train_Y.type(torch.Tensor)
    workers_config = copy.deepcopy(config)
    for k in range(config['num_partitions']):
            s = n // config['num_partitions']
            x, y = train_X[k*s:(k+1)*s], train_Y[k*s:(k+1)*s]
            workers.append(Worker(x,y,workers_config))

    # Run algorithms
    for algorithm in algorithms:
        if algorithm == 'GIANT':
            GIANT(workers, copy.deepcopy(config), test_X, test_Y)

        if algorithm == 'DINGO':
            DINGO(workers, copy.deepcopy(config), test_X, test_Y)

        if algorithm == 'DISCO':
            DISCO(workers, copy.deepcopy(config), test_X, test_Y)

        if algorithm == 'InexactDANE':
            inexactDANE(workers, copy.deepcopy(config), test_X, test_Y)

        if algorithm == 'AIDE':
            AIDE(workers, copy.deepcopy(config), test_X, test_Y)

        if algorithm == 'Asynchronous SGD':
            SGD(workers, copy.deepcopy(config), 'asynchronous', test_X, test_Y)

        if algorithm == 'Synchronous SGD':
            SGD(workers, copy.deepcopy(config), 'synchronous', test_X, test_Y)


if __name__ == "__main__":
    main()
