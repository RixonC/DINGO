from DINGO import DINGO
import torch
import torch.distributed as dist


def main():
    test_accuracy = True # compute and plot test accuracy or error

    torch.set_default_tensor_type(torch.FloatTensor)

    seed = 1
    torch.manual_seed(seed)

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

    if data == 'EMNIST_digits':
        n = 240000
        p = 784
        classes = 10
    elif data == 'CIFAR10':
        n = 50000
        p = 3072
        classes = 10
    else:
        n = 20000
        p = 784
        classes = 10

    autoencoder_layers = None
    if objective_function == 'Autoencoder':
        autoencoder_layers = [p,400,300,200,100,50,25,12,6,12,25,50,100,200,300,400,p]
        dimension = sum([autoencoder_layers[k+1]*(autoencoder_layers[k]+1) for k in range(len(autoencoder_layers)-1)])
    elif objective_function == 'softmax':
        dimension = p*(classes-1)

    num_partitions = int(dist.get_world_size()-1) # number of workers

    config = {'num_partitions' : num_partitions,
              'max_iterations' : 10000,
              'max_communication_rounds' : 500,
              'grad_tol' : 1e-8, # if norm(gradient)<grad_tol then algorithm loop breaks
              'lamda' : 1.0/n, # regularization parameter
              'subproblem_max_iterations' : 50, # maximum iterations for CG, MINRES-QLP and LSMR
              'subproblem_tolerance' : 1e-4, # relative residual tolerance in CG, MINRES-QLP and LSMR
              'use_preconditioning_in_subproblem' : False,
              'line_search_max_iterations' : 50,
              'line_search_factor' : 0.5, # we perform backtracking line search with the step-sizes {1, line_search_factor, line_search_factor^2, ..., line_search_factor ^ line_search_max_iterations}.
              'line_search_rho' : 1e-4, # Armijo line-search parameter.
              'DINGO_theta' : 1e-4,
              'DINGO_phi' : 1e-6,
              'obj_fun' : objective_function,
              'dimension' : dimension,
              'classes' : classes,
              'features' : p,
              'local_sample_size' : int(float(n)/num_partitions),
              'autoencoder_layers' : autoencoder_layers,
              'dataset' : data,
              'w0_str' : starting_point,
              'test_accuracy' : test_accuracy}

    if dist.get_rank() == 0:
        print("\nDINGO\n")
        print("Dataset: " + config['dataset'])
        print("Whole Training Dataset Size: ({},{})".format(n, config['features']))
        print("Number of Workers: {}".format(num_partitions))
        print("Workers' Local Sample Size: {}".format(config['local_sample_size']))
        print("Objective Function: " + objective_function)
        print("Problem Dimension: {}".format(dimension))
        print("Maximum Iterations: {}".format(config['max_iterations']))
        print("Maximum Communication Rounds: {}".format(config['max_communication_rounds']))
        print("Gradient Norm Tolerance: {}".format(config['grad_tol']))
        print("Regularization Parameter: {}".format(config['lamda']))
        print("Sub-Problem Solver Maximum Iterations: {}".format(config['subproblem_max_iterations']))
        print("Sub-Problem Solver Tolerance: {}".format(config['subproblem_tolerance']))
        print("Line-Search Maximum Iterations: {}".format(config['line_search_max_iterations']))
        print("Line-Search Factor: {}".format(config['line_search_factor']))
        print("Armijo Line-Search Parameter: {}".format(config['line_search_rho']))
        print("DINGO theta: {}".format(config['DINGO_theta']))
        print("DINGO phi: {}".format(config['DINGO_phi']))
        print("Starting Point: " + config['w0_str'])
        if objective_function == 'Autoencoder':
            print("Autoencoder Layers: " + str(autoencoder_layers))

    DINGO(config)


if __name__ == "__main__":
    dist.init_process_group('mpi')
    main()
