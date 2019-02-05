# -*- coding: utf-8 -*-

from __future__ import division

import copy
from resultsRecorder import ResultsRecorder
from time import time
import torch


def distributed_PCG_algorithm(workers, config, test_X=None, test_Y=None):
    '''Perform the distributed PCG algorithm.

    Args:
        workers: A list of worker class instances.
        config: A dictionary of all necessary parameters.
        test_X: test features.
        test_Y: test labels.
    '''
    weights = config['w0']

    gradient = (1/config['num_partitions']) * sum([W.grad(weights) for W in workers]) # [d,1]
    d = len(gradient)

    # initialization
    v_t = torch.zeros((d,1)) # [d,1]
    r_t = gradient # [d,1]
    if config['use_preconditioning_in_subproblem']:
        worker1 = workers[0]
        s_t = worker1.DISCO_P_inv_vect(weights, r_t)[0] # [d,1]
    else:
        s_t = r_t # [d,1]
    u_t = s_t # [d,1]

    # repeat
    T = 0
    while T < config['subproblem_max_iterations']:
        hess_times_u_t = (1/config['num_partitions']) * sum([W.hess_vect(weights, u_t) for W in workers]) # [d,1]
        hess_times_v_t = (1/config['num_partitions']) * sum([W.hess_vect(weights, v_t) for W in workers]) # [d,1]

        if config['use_preconditioning_in_subproblem'] == False:
            if torch.norm(hess_times_v_t-gradient)/torch.norm(gradient) <= config['subproblem_tolerance']:
                break
        else:
            P_inv_hess_times_v_t = worker1.DISCO_P_inv_vect(weights, hess_times_v_t)[0]
            P_inv_gradient = worker1.DISCO_P_inv_vect(weights, gradient)[0]
            if torch.norm(P_inv_hess_times_v_t-P_inv_gradient)/torch.norm(P_inv_gradient) <= config['subproblem_tolerance']:
                break

        uHu = torch.mm(u_t.transpose(0,1), hess_times_u_t)
        if uHu <= 0:
            return None # failed to recover a solution

        alpha_t = torch.mm(r_t.transpose(0,1), s_t) / uHu # [1]
        v_t_plus_1 = v_t + alpha_t * u_t # [d,1]
        r_t_plus_1 = r_t - alpha_t * hess_times_u_t # [d,1]

        if config['use_preconditioning_in_subproblem']:
            s_t_plus_1 = worker1.DISCO_P_inv_vect(weights, r_t_plus_1)[0] # [d,1]
        else:
            s_t_plus_1 = r_t_plus_1 # [d,1]

        beta_t = torch.mm(r_t_plus_1.transpose(0,1), s_t_plus_1) / torch.mm(r_t.transpose(0,1), s_t) # [1]
        u_t_plus_1 = s_t_plus_1 + beta_t * u_t # [d,1]

        v_t = v_t_plus_1 # [d,1]
        r_t = r_t_plus_1 # [d,1]
        s_t = s_t_plus_1 # [d,1]
        u_t = u_t_plus_1 # [d,1]
        T += 1

    v_k = v_t_plus_1 # [d,1]
    r_k = r_t_plus_1 # [d,1]
    delta_k = torch.sqrt(torch.mm(v_k.transpose(0,1), hess_times_v_t) + alpha_t * torch.mm(v_k.transpose(0,1), hess_times_u_t)) # [1]
    return v_k, r_k, delta_k, T


def DISCO(workers, config, test_X=None, test_Y=None):
    '''Perform the DiSCO algorithm.

    Args:
        workers: A list of worker class instances.
        config: A dictionary of all necessary parameters.
        test_X: test features.
        test_Y: test labels.
    '''
    print('\n #############################################################################')
    print('\n DiSCO \n')

    # if the subproblem fails then the algorithm stops
    subproblem_failed_end_message = 'uHu<=0 in PCG'

    # the Recorder class records, plots and saves results
    Recorder = ResultsRecorder('DISCO', config, test_X, test_Y)

    if config['load_plot_data']:
        PlotData = Recorder.get_plot_data()
    else:
        PlotData = None

    if PlotData != None: # there is a saved run that matches our current configuration
        final_weights, end_message, lists = Recorder.get_plot_data()
        iterations = len(lists['cumulative_communication_rounds_list'])
        for k in range(iterations):
            Recorder.print_row(k, lists)
        Recorder.print_row(iterations, lists, final_row=True)
        print('\n     ' + end_message)
        Recorder.print_plots(end_message, lists, subproblem_failed_end_message)
        # return the final: weights/point, loss, norm of gradient, and test accuracy or error
        return final_weights, lists['loss_list'][-1], lists['grad_norm_list'][-1], lists['test_accuracy_list'][-1]
    else: # run the experiment
        weights = config['w0']

        # these return the loss (function value) and gradient of the function
        get_loss = lambda: (1/config['num_partitions']) * sum([W.loss(weights) for W in workers]) # []
        get_grad = lambda: (1/config['num_partitions']) * sum([W.grad(weights) for W in workers]) # [d,1]

        # results are added to these lists
        lists = {'cumulative_communication_rounds_list' : [],
                 'loss_list' : [],
                 'grad_norm_list' : [],
                 'PCG_iters_list' : [],
                 'delta_list' : [],
                 'time_list' : [],
                 'test_accuracy_list' : []}

        # we will store a message about why the algorithm stopped
        end_message = 'max_iterations reached'

        iteration = 0
        total_communication_rounds = 0
        while iteration < config['max_iterations']:
            if total_communication_rounds >= config['max_communication_rounds']:
                end_message = 'max_communication_rounds reached'
                break
            start_time = time()

            gradient = get_grad()
            grad_norm = torch.norm(gradient)
            if grad_norm <= config['grad_tol']:
                end_message = 'grad_tol reached'
                break

            distributed_PCG_algorithm_config = copy.deepcopy(config)
            distributed_PCG_algorithm_config['w0'] = weights

            PCG_result = distributed_PCG_algorithm(workers, distributed_PCG_algorithm_config)
            if PCG_result == None: # distributed_PCG_algorithm failed to recover a solution
                end_message = subproblem_failed_end_message
                break

            lists['grad_norm_list'].append(grad_norm)
            lists['loss_list'].append(get_loss())
            lists['test_accuracy_list'].append(Recorder.record_accuracy(weights)) # record test accuracy or error

            v_k, r_k, delta_k, PCG_iters = PCG_result
            lists['PCG_iters_list'].append(PCG_iters)
            lists['delta_list'].append(delta_k)

            # each iteration of DiSCO uses 2+2*PCG_iters rounds of communication
            total_communication_rounds += 2+2*PCG_iters
            lists['cumulative_communication_rounds_list'].append(total_communication_rounds)

            weights = weights-(1/(1+delta_k))*v_k
            lists['time_list'].append(time() - start_time)
            Recorder.print_row(iteration, lists) # print the current row of the results table
            iteration += 1

        lists['loss_list'].append(get_loss())
        lists['grad_norm_list'].append(torch.norm(get_grad()))
        lists['test_accuracy_list'].append(Recorder.record_accuracy(weights))
        Recorder.print_row(iteration, lists, final_row=True) # print the last row of the results table

        # print why the algorithm stopped
        print('\n     ' + end_message + ' in {:.2f} seconds'.format(sum(lists.get('time_list'))))

        if config['save_plot_data']:
            Recorder.save_plots_data(weights, end_message, lists)
        
        Recorder.print_plots(end_message, lists, subproblem_failed_end_message) # generate and save plots

        # return the final: weights/point, loss, norm of gradient, and test accuracy or error
        return weights, lists['loss_list'][-1], lists['grad_norm_list'][-1], lists['test_accuracy_list'][-1]
