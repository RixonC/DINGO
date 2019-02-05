from myLoadData import myDataLoader
import numpy as np
from resultsRecorder import ResultsRecorder
from time import time
import torch
import torch.distributed as dist
from worker import Worker


def get_initial_weights(config):
    '''Return the initial weight vector.

    Args:
        config: A dictionary of all necessary parameters.
    '''
    if config['w0_str'] == 'zeros':
        w0 = torch.zeros((config['dimension'],1))
    elif config['w0_str'] == 'ones':
        w0 = torch.ones((config['dimension'],1))
    else:
        if config['obj_fun'] == 'Autoencoder': # default random initialization of PyTorch
            ws = []
            autoencoder_layers = config['autoencoder_layers']
            for k in range(len(autoencoder_layers)-1):
                M = torch.zeros((autoencoder_layers[k+1]*autoencoder_layers[k],1))
                b = torch.zeros((autoencoder_layers[k+1],1))
                bound = 1.0/np.sqrt(autoencoder_layers[k])
                ws.append(M.uniform_(-bound,bound))
                ws.append(b.uniform_(-bound,bound))
            w0 = torch.cat(ws, dim=0)
        else:
            w0 = torch.randn(config['dimension'],1)
    return w0


def get_worker(rank, config):
    '''Return a Worker class instance, which is initialized on the subset of data corresponding to the process.

    Args:
        rank: The process rank.
        config: A dictionary of all necessary parameters.
    '''
    if rank == 0:
        '''
        For simplicity, the driver loads, processes and partitions the dataset and then scatters the partitions across the workers.
        In practice, each worker could load their data from local storage.
        '''
        DL = myDataLoader(config) # the myDataLoader class loads and processes data
        train_X, train_Y = DL.load_train_data()
        n, p = train_X.shape
        s = int(float(n) / config['num_partitions'])

        train_Xs = [torch.zeros(train_X[0:s].shape).contiguous()] # the first element is sent on the driver
        for k in range(config['num_partitions']):
            train_Xs.append(train_X[(k)*s:(k+1)*s].contiguous())
        dist.scatter(tensor=torch.zeros(train_X[0:s].shape).contiguous(), scatter_list=train_Xs, src=0)

        if config['obj_fun'] == 'softmax':
            train_Ys = [torch.zeros(train_Y[0:s].shape).contiguous()] # the first element is sent on the driver
            for k in range(config['num_partitions']):
                train_Ys.append(train_Y[(k)*s:(k+1)*s].contiguous())
            dist.scatter(tensor=torch.zeros(train_Y[0:s].shape).contiguous(), scatter_list=train_Ys, src=0)
        return None

    if rank > 0:
        train_X = torch.zeros(config['local_sample_size'], config['features'])
        dist.scatter(tensor=train_X, scatter_list=[], src=0)
        if config['obj_fun'] == 'softmax':
            train_Y = torch.zeros(config['local_sample_size'], config['classes']-1)
            dist.scatter(tensor=train_Y, scatter_list=[], src=0)
        else:
            train_Y = None
        return Worker(train_X, train_Y, config)


def compute_local_line_search_matrix(worker, weights, direction, config):
    '''Return a matrix where column k is the local objective value and gradient at the point: weights + config['line_search_factor']^k * direction.

    Args:
        worker: A Worker class instance.
        weights: The current point.
        direction: The update direction.
        config: A dictionary of all necessary parameters.
    '''
    local_loss_and_gradients = []
    k = 0
    alpha = 1
    while k < 1 + config['line_search_max_iterations']:
        temp_weights = weights + alpha * direction
        local_loss_and_gradients.append(torch.cat([worker.loss(temp_weights).reshape([1,1]), worker.grad(temp_weights)], dim=0))
        k += 1
        alpha *= config['line_search_factor']
    return torch.cat(local_loss_and_gradients, dim=1)


def compute_line_search_alpha_that_passes(line_search_matrix, direction_dot_hess_times_grad, gradient_norm_squared, config):
    '''Return the largest step-size that passes backtracking line search on the norm of the gradient squared. Otherwise, return the smallest step-size.

    Args:
        line_search_matrix: A matrix where column k is the objective value and gradient at the point: weights + config['line_search_factor']^k * direction.
        direction_dot_hess_times_grad: The inner product of the update direction and the Hessian-gradient product.
        gradient_norm_squared: The norm of the gradient squared.
        config: A dictionary of all necessary parameters.
    '''
    k = 0
    alpha = 1
    gradient_at_new_weights = line_search_matrix[1:,k:k+1]
    while torch.mm(gradient_at_new_weights.transpose(0,1), gradient_at_new_weights) \
            > gradient_norm_squared + 2*alpha*config['line_search_rho']*direction_dot_hess_times_grad \
            and k < config['line_search_max_iterations']:
        k += 1
        alpha *= config['line_search_factor']
        gradient_at_new_weights = line_search_matrix[1:,k:k+1] # column k
    return alpha, k, line_search_matrix[:,k:k+1]


def DINGO(config):
    '''Perform the DINGO algorithm.

    Args:
        config: A dictionary of all necessary parameters.
    '''
    rank = dist.get_rank()
    
    if rank == 0:        
        # the driver constructs the initial weights
        weights = get_initial_weights(config)
        # the Recorder class computes test accuracy or error, and prints and plots results of DINGO
        if config['test_accuracy']:
            test_X, test_Y = myDataLoader(config).load_test_data()
            Recorder = ResultsRecorder(config, test_X, test_Y)
        else:
            Recorder = ResultsRecorder(config)
        # we will store a message about why DINGO stopped
        end_message = 'max_iterations reached'
        # results are added to these lists
        lists = {'cumulative_communication_rounds_list' : [],
                 'loss_list' : [],
                 'grad_norm_list' : [],
                 'avg_Lagrangian_lambda_list' : [],
                 'alpha_list' : [],
                 'ls_exp_list' : [],
                 'direction_norm_list' : [],
                 'inner_prod_list' : [],
                 'time_list' : [],
                 'cases_list' : [],
                 'test_accuracy_list' : []}

    # all workers initialize a class that contains their dataset and computation instructions
    worker = get_worker(rank, config)

    # all processes count the number of iterations and communication rounds
    iteration = 0
    total_communication_rounds = 0
    while iteration < config['max_iterations']:
        if total_communication_rounds >= config['max_communication_rounds']:
            end_message = 'max_communication_rounds reached'
            break

        ###############################################################################################
        #--------------------------- compute objective value and gradient ----------------------------#

        if rank == 0:
            # the driver will record how long each iteration takes
            start_time = time()

        if iteration == 0:
            # this iteration requires an additional two communication rounds

            if rank == 0:
                # in subsequent iterations, the driver only needs to broadcast alpha to update weights
                alpha = 0
                # the driver broadcasts the weights to all workers
                dist.broadcast(weights, 0)
                total_communication_rounds += 1

            if rank > 0:
                # in subsequent iterations, the driver only needs to broadcast alpha to update weights
                direction = 0
                # all workers receive the initial weights
                weights = torch.zeros(config['dimension'], 1)
                dist.broadcast(weights, 0)
                total_communication_rounds += 1

                # all workers compute their local objective value and gradient
                local_loss = worker.loss(weights).reshape([1,1])
                local_gradient = worker.grad(weights)
                local_loss_and_gradient = torch.cat([local_loss, local_gradient], dim=0)

                # all workers send their local objective value and gradient to the driver
                dist.reduce(local_loss_and_gradient, 0)
                total_communication_rounds += 1

            if rank == 0:
                # the driver receives and averages all workers' local objective value and gradient
                loss_and_gradient = torch.zeros(1+config['dimension'], 1)
                dist.reduce(loss_and_gradient, 0)
                total_communication_rounds += 1
                loss_and_gradient = (1.0/config['num_partitions']) * loss_and_gradient

        if rank == 0:
            loss = loss_and_gradient[0:1]
            gradient = loss_and_gradient[1:]
            gradient_norm = torch.norm(gradient)
            # the driver broadcasts the step-size and gradient to all workers
            dist.broadcast(torch.cat([torch.tensor(float(alpha)).reshape([1,1]), gradient], dim=0), 0)
            total_communication_rounds += 1
            if gradient_norm <= config['grad_tol']:
                end_message = 'grad_tol reached'
                break
            gradient_norm_squared = gradient_norm.pow(2)
            lists['loss_list'].append(loss)
            lists['grad_norm_list'].append(gradient_norm)
            lists['test_accuracy_list'].append(Recorder.compute_test_accuracy(weights)) # record test accuracy or error

        ###############################################################################################
        #------ compute Hessian-gradient product, and Case 1, Case 2 or Case 3 update direction ------#

        if rank > 0:
            # all workers receive the step-size and gradient
            alpha_and_gradient = torch.zeros(1+config['dimension'], 1)
            dist.broadcast(alpha_and_gradient, 0)
            total_communication_rounds += 1
            alpha = alpha_and_gradient[0:1]
            gradient = alpha_and_gradient[1:]
            gradient_norm = torch.norm(gradient)
            if gradient_norm <= config['grad_tol']:
                break
            gradient_norm_squared = gradient_norm.pow(2)
            # all workers update to current weights
            weights = weights + alpha * direction

            local_hess_times_grad = worker.hess_vect(weights, gradient)
            local_hess_inv_grad = worker.hess_inv_vect(weights, gradient, method='MINRES-QLP')[0]
            local_hess_tilde_inv_grad_tilde = worker.hess_tilde_inv_grad_tilde(weights, gradient)[0]
            local_Hg_C1_C2_stack = torch.cat([local_hess_times_grad, local_hess_inv_grad, local_hess_tilde_inv_grad_tilde], dim=1)
            dist.gather(tensor=local_Hg_C1_C2_stack, gather_list=[], dst=0)
            total_communication_rounds += 1

        if rank == 0:
            local_Hg_C1_C2_stacks = [torch.zeros(config['dimension'], 3) for _ in range(1+config['num_partitions'])]
            dist.gather(tensor=torch.zeros(config['dimension'], 3), gather_list=local_Hg_C1_C2_stacks, dst=0)
            total_communication_rounds += 1
            local_Hg_C1_C2_stacks.pop(0) # the first element is a zero matrix from the driver

            hess_times_grad = (1.0/config['num_partitions']) * sum([T[:,0:1] for T in local_Hg_C1_C2_stacks])
            hess_inv_grad_list = [T[:,1:2] for T in local_Hg_C1_C2_stacks]
            p = (-1.0/config['num_partitions']) * sum(hess_inv_grad_list) # Case 1 update direction

            if torch.mm(p.transpose(0,1), hess_times_grad) <= -config['DINGO_theta']*gradient_norm_squared: # try Case 1
                direction = p
                lists['cases_list'].append(1)
                lists['avg_Lagrangian_lambda_list'].append(0)
            else: # Case 1 failed
                hess_tilde_inv_grad_tilde_list = [T[:,2:3] for T in local_Hg_C1_C2_stacks]
                p = (-1.0/config['num_partitions']) * sum(hess_tilde_inv_grad_tilde_list) # Case 2 update direction
                if torch.mm(p.transpose(0,1), hess_times_grad) <= -config['DINGO_theta']*gradient_norm_squared: # try Case 2
                    direction = p
                    lists['cases_list'].append(2)
                    lists['avg_Lagrangian_lambda_list'].append(0)
                else: # Case 1 and Case 2 failed, so now do Case 3
                    '''
                    For simplicity, the driver broadcasts the Hessian-gradient product to all workers.
                    In practice, the Hessian-gradient product only needs to be sent to the workers that need to compute the local Case 3 update direction.
                    '''
                    lists['cases_list'].append(3)
                    # the 1 indicates to the worker that it needs to compute the local Case 3 update direction
                    cat_1_and_hess_times_grad = torch.cat([torch.ones((1,1)), hess_times_grad], dim=0)
                    dist.broadcast(cat_1_and_hess_times_grad, 0)
                    total_communication_rounds += 1

                    # the driver computes the Case 3 update direction
                    lagrangian_lambda_and_direction = torch.zeros(1+config['dimension'], 1)
                    dist.reduce(lagrangian_lambda_and_direction, 0)
                    total_communication_rounds += 1
                    lagrangian_lambda_and_direction = (1.0/config['num_partitions']) * lagrangian_lambda_and_direction
                    avg_lagrangian_lambda = lagrangian_lambda_and_direction[0:1]
                    lists['avg_Lagrangian_lambda_list'].append(avg_lagrangian_lambda)
                    direction = lagrangian_lambda_and_direction[1:]

            direction_norm = torch.norm(direction)
            lists['direction_norm_list'].append(direction_norm)
            cat_minus_1_and_hess_times_grad = torch.cat([-1*torch.ones((1,1)), direction], dim=0)
            dist.broadcast(cat_minus_1_and_hess_times_grad, 0)
            total_communication_rounds += 1

        ###############################################################################################
        #---------------------------------------- Line Search ----------------------------------------#

        if rank > 0:
            stack = torch.zeros(1+config['dimension'], 1)
            dist.broadcast(stack, 0)
            total_communication_rounds += 1

            if stack[0:1] < 0:
                direction = stack[1:]
                local_line_search_matrix = compute_local_line_search_matrix(worker, weights, direction, config)
                dist.reduce(local_line_search_matrix, 0)
                total_communication_rounds += 1

            else:
                hess_times_grad = stack[1:]
                local_lagrangian_direction, local_lagrangian_lambda \
                    = worker.lagrangian_direction(weights, gradient, hess_times_grad, local_hess_tilde_inv_grad_tilde)
                local_lagrangian_lambda.reshape([1,1])
                local_lagrangian_lambda_and_direction = torch.cat([local_lagrangian_lambda, local_lagrangian_direction], dim=0)
                dist.reduce(local_lagrangian_lambda_and_direction, 0)
                total_communication_rounds += 1

                stack = torch.zeros(1+config['dimension'], 1)
                dist.broadcast(stack, 0)
                total_communication_rounds += 1
                assert(stack[0:1] < 0)
                direction = stack[1:]
                local_line_search_matrix = compute_local_line_search_matrix(worker, weights, direction, config)
                dist.reduce(local_line_search_matrix, 0)
                total_communication_rounds += 1

        if rank == 0:
            # the driver receives and averages all workers' local line-search matrix
            line_search_matrix = torch.zeros(1+config['dimension'], 1+config['line_search_max_iterations'])
            dist.reduce(line_search_matrix, 0)
            total_communication_rounds += 1
            line_search_matrix = (1.0/config['num_partitions']) * line_search_matrix

            direction_dot_hess_times_grad = torch.mm(direction.transpose(0,1), hess_times_grad)
            lists['inner_prod_list'].append(direction_dot_hess_times_grad)
            alpha, k, loss_and_gradient_at_new_weights \
                = compute_line_search_alpha_that_passes(line_search_matrix, direction_dot_hess_times_grad, gradient_norm_squared, config)
            lists['alpha_list'].append(alpha)
            lists['ls_exp_list'].append(k)

            weights = weights + alpha * direction
            loss_and_gradient = loss_and_gradient_at_new_weights

            lists['cumulative_communication_rounds_list'].append(total_communication_rounds)
            lists['time_list'].append(time()-start_time)
            Recorder.print_row(iteration, lists) # print the current row of the results table

        iteration += 1

    if rank == 0:
        lists['loss_list'].append(loss_and_gradient[0:1])
        lists['grad_norm_list'].append(torch.norm(loss_and_gradient[1:]))
        lists['test_accuracy_list'].append(Recorder.compute_test_accuracy(weights))
        Recorder.print_row(iteration, lists, final_row=True) # print the final row of the results table
        print('\n     ' + end_message + ' in {:.2f} seconds'.format(sum(lists['time_list']))) # print why DINGO stopped
        Recorder.print_plots(lists) # generate and save plots
