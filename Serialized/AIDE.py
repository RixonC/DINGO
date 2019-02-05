# -*- coding: utf-8 -*-

from __future__ import division

import copy
from inexactDANE import inexactDANE_algorithm
from math import sqrt
from resultsRecorder import ResultsRecorder
from time import time
import torch


def AIDE(workers, config, test_X=None, test_Y=None):
    '''Perform the AIDE algorithm.

    Args:
        workers: A list of worker class instances.
        config: A dictionary of all necessary parameters.
        test_X: test features.
        test_Y: test labels.
    '''
    print('\n #######################################################')
    print('\n AIDE \n')

    # the Recorder class records, plots and saves results
    Recorder = ResultsRecorder('AIDE', config, test_X, test_Y)

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
        # return the final: weights/point, loss, norm of gradient, and test accuracy or error
        return final_weights, lists['loss_list'][-1], lists['grad_norm_list'][-1], lists['test_accuracy_list'][-1]
    else: # run the experiment
        inexactDANE_algorithm_config = copy.deepcopy(config)
        inexactDANE_algorithm_config['max_iterations'] = 1 # one iteration of AIDE will call one iteration of InexactDANE
        inexactDANE_algorithm_config['print_results'] = False

        weights = config['w0']
        y = config['w0']
        zeta = 0

        if config['lamda'] == 0:
            lamda = 1/(10*config['num_partitions'])
            q = lamda/(lamda + config['AIDE_tau'])
        else:
            q = config['lamda']/(config['lamda'] + config['AIDE_tau'])

        # results are added to these lists
        lists = {'loss_list' : [],
                 'grad_norm_list' : [],
                 'time_list' : [],
                 'cumulative_communication_rounds_list' : [],
                 'InexactDANE_iterations' : [],
                 'test_accuracy_list' : []}

        # we will store a message about why the algorithm stopped
        end_message = 'max_iterations reached'

        # these return the loss (function value) and gradient of the original function
        get_loss = lambda: (1/config['num_partitions']) * sum([W.loss(weights) for W in workers]) # []
        get_grad = lambda: (1/config['num_partitions']) * sum([W.grad(weights) for W in workers]) # [d,1]

        iteration = 0
        total_communication_rounds = 0
        while iteration < config['max_iterations']:
            if total_communication_rounds >= config['max_communication_rounds']:
                end_message = 'max_communication_rounds reached'
                break
            start_time = time()

            grad_norm = torch.norm(get_grad())
            if grad_norm <= config['grad_tol']:
                end_message = 'grad_tol reached'
                break
            lists['grad_norm_list'].append(grad_norm)
            lists['loss_list'].append(get_loss())
            lists['test_accuracy_list'].append(Recorder.record_accuracy(weights)) # record test accuracy or error

            inexactDANE_algorithm_config['w0'] = weights
            inexactDANE_algorithm_config['AIDE_y'] = y
            new_weights, t, c, _ = inexactDANE_algorithm(workers, inexactDANE_algorithm_config)
            total_communication_rounds += c # all rounds of communication are in the InexactDANE algorithm
            lists['cumulative_communication_rounds_list'].append(total_communication_rounds)

            b = zeta**2 - q
            new_zeta = (-b + sqrt(b**2 + 4*(zeta**2)))/2
            if new_zeta < 0 or new_zeta > 1:
                new_zeta = (-b - sqrt(b**2 + 4*(zeta**2)))/2
            assert new_zeta > 0 and new_zeta < 1

            beta = (zeta*(1-zeta))/(zeta**2+new_zeta)
            y = new_weights + beta*(new_weights-weights)
            weights = new_weights
            zeta = new_zeta
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
            Recorder.save_plots_data(weights, end_message, lists) # saves the results of this run
        
        Recorder.print_plots(end_message, lists) # generate and save plots

        # return the final: weights/point, loss, norm of gradient, and test accuracy or error
        return weights, lists['loss_list'][-1], lists['grad_norm_list'][-1], lists['test_accuracy_list'][-1]
