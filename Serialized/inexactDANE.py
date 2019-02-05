# -*- coding: utf-8 -*-

from __future__ import division

from resultsRecorder import ResultsRecorder
from time import time
import torch


def inexactDANE_algorithm(workers, config, test_X=None, test_Y=None):
    '''Perform the standalone InexactDANE algorithm.

    Args:
        workers: A list of worker class instances.
        config: A dictionary of all necessary parameters.
        test_X: test features.
        test_Y: test labels.
    '''
    weights = config['w0']

    # these return the loss (function value) and gradient of the function
    get_loss = lambda: (1/config['num_partitions']) * sum([W.loss(weights, AIDE_tau=config['AIDE_tau'], AIDE_y=config['AIDE_y']) for W in workers]) # []
    get_grad = lambda: (1/config['num_partitions']) * sum([W.grad(weights, AIDE_tau=config['AIDE_tau'], AIDE_y=config['AIDE_y']) for W in workers]) # [d,1]

    if config['print_results']:
        # the Recorder class records, plots and saves results
        Recorder = ResultsRecorder('InexactDANE', config, test_X, test_Y)

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
            Recorder.print_plots(end_message, lists)
            # return the final weights/point, number of iterations, number of communication rounds, and the results lists
            return final_weights, len(lists['cumulative_communication_rounds_list'])-1, lists['cumulative_communication_rounds_list'][-1], lists

        # results are added to these lists
        lists = {'loss_list' : [],
                 'grad_norm_list' : [],
                 'cumulative_communication_rounds_list' : [],
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
        total_communication_rounds += 2

        if config['print_results']:
            lists['grad_norm_list'].append(grad_norm)
            lists['loss_list'].append(get_loss())
            lists['test_accuracy_list'].append(Recorder.record_accuracy(weights)) # record test accuracy or error

        # each worker inexactly solves local DANE subproblem and then we average solutions
        weights = (1/config['num_partitions']) * sum([W.inexactDANE_subproblem(weights, gradient, AIDE_tau=config['AIDE_tau'], AIDE_y=config['AIDE_y']) for W in workers]) # [d,1]
        total_communication_rounds += 2

        if config['print_results']:
            lists['cumulative_communication_rounds_list'].append(total_communication_rounds)
            lists['time_list'].append(time() - start_time)
            Recorder.print_row(iteration, lists) # print the current row of the results table
        
        iteration += 1

    if config['print_results']:
        lists['loss_list'].append(get_loss())
        lists['grad_norm_list'].append(torch.norm(get_grad()))
        lists['test_accuracy_list'].append(Recorder.record_accuracy(weights))
        Recorder.print_row(iteration, lists, final_row=True) # print the last row of the results table

        # print why the algorithm stopped
        print('\n     ' + end_message + ' in {:.2f} seconds'.format(sum(lists['time_list'])))

        if config['save_plot_data']:
            Recorder.save_plots_data(weights, end_message, lists)
        
        Recorder.print_plots(end_message, lists) # generate and save plots
        return weights, iteration, total_communication_rounds, lists

    # return the final weights/point, number of iterations, number of communication rounds, and the results lists
    return weights, iteration, total_communication_rounds, None


def inexactDANE(workers, config, test_X=None, test_Y=None):
    '''Perform the InexactDANE algorithm.

    Args:
        workers: A list of worker class instances.
        config: A dictionary of all necessary parameters.
        test_X: test features.
        test_Y: test labels.
    '''

    print('\n #######################################################')
    print('\n InexactDANE \n')

    config['print_results'] = True
    config['AIDE_tau'] = 0
    config['AIDE_y'] = 0
    weights, _, _, lists = inexactDANE_algorithm(workers, config, test_X, test_Y)

    # return the final: weights/point, loss, norm of gradient, and test accuracy or error
    return weights, lists['loss_list'][-1], lists['grad_norm_list'][-1], lists['test_accuracy_list'][-1]
