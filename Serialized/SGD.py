# -*- coding: utf-8 -*-

from __future__ import division

from resultsRecorder import ResultsRecorder
import torch
from time import time


def SGD(workers, config, SGD_type, test_X=None, test_Y=None):
    '''Perform the Async-SGD or Sync-SGD algorithm.

    Args:
        workers: A list of worker class instances.
        config: A dictionary of all necessary parameters.
        SGD_type: Async-SGD or Sync-SGD.
        test_X: test features.
        test_Y: test labels.
    '''
    print('\n ##################################################################')
    if SGD_type == 'synchronous':
        print('\n Synchronous SGD: alpha = {}, min-batch_size = {} \n'.format(config['Synchronous_SGD_stepsize'], config['SGD_minibatch_size']))
        # the Recorder class records, plots and saves results
        Recorder = ResultsRecorder('Synchronous SGD', config, test_X, test_Y)
    else:
        print('\n Asynchronous SGD: alpha = {}, min-batch_size = {} \n'.format(config['Asynchronous_SGD_stepsize'], config['SGD_minibatch_size']))
        # the Recorder class records, plots and saves results
        Recorder = ResultsRecorder('Asynchronous SGD', config, test_X, test_Y)

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
        weights = config['w0']

        # these return the loss (function value), full gradient and local mini-batch gradients of the function
        get_loss = lambda: (1/config['num_partitions']) * sum([W.loss(weights) for W in workers]) # []
        get_grad = lambda: (1/config['num_partitions']) * sum([W.grad(weights) for W in workers]) # [d,1]
        get_minibatch_grad_list = lambda: [W.minibatch_grad(weights) for W in workers]

        # results are added to these lists
        lists = {'cumulative_communication_rounds_list' : [],
                 'loss_list' : [],
                 'grad_norm_list' : [],
                 'direction_norm_list' : [],
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

            grad_norm = torch.norm(get_grad())
            if grad_norm <= config['grad_tol']:
                end_message = 'grad_tol reached'
                break
            lists['grad_norm_list'].append(grad_norm)
            lists['loss_list'].append(get_loss())
            lists['test_accuracy_list'].append(Recorder.record_accuracy(weights)) # record test accuracy or error

            if SGD_type == 'synchronous':
                minibatch_grad_list = get_minibatch_grad_list()
                direction = -(1/config['num_partitions']) * sum(g[0] for g in minibatch_grad_list) # [d,1]
                weights += config['Synchronous_SGD_stepsize'] * direction
            else:
                if iteration == 0:
                    minibatch_grad_list = get_minibatch_grad_list()
                sorted_minibatch_grad_list = sorted(enumerate(minibatch_grad_list), key=lambda x:x[1][1]) # (worker number, minibatch_grad tuple)
                direction = -sorted_minibatch_grad_list[0][1][0]
                weights += config['Asynchronous_SGD_stepsize'] * direction
                worker_num = sorted_minibatch_grad_list[0][0]
                worker_time = sorted_minibatch_grad_list[0][1][1]
                replacement_minibatch_grad = workers[worker_num].minibatch_grad(weights)
                replacement_minibatch_grad = (replacement_minibatch_grad[0], replacement_minibatch_grad[1] + worker_time)
                minibatch_grad_list[worker_num] = replacement_minibatch_grad

            total_communication_rounds += 2
            lists['direction_norm_list'].append(torch.norm(direction))
            lists['cumulative_communication_rounds_list'].append(total_communication_rounds)
            lists['time_list'].append(time()-start_time)
            Recorder.print_row(iteration, lists) # print the current row of the results table
            iteration += 1

        lists['grad_norm_list'].append(torch.norm(get_grad()))
        lists['loss_list'].append(get_loss())
        lists['test_accuracy_list'].append(Recorder.record_accuracy(weights))
        Recorder.print_row(iteration, lists, final_row=True) # print the last row of the results table

        # print why the algorithm stopped
        print('\n     ' + end_message + ' in {:.2f} seconds'.format(sum(lists['time_list'])))

        if config['save_plot_data']:
            Recorder.save_plots_data(weights, end_message, lists)
        
        Recorder.print_plots(end_message, lists) # generate and save plots

        # return the final: weights/point, loss, norm of gradient, and test accuracy or error
        return weights, lists['loss_list'][-1], lists['grad_norm_list'][-1], lists['test_accuracy_list'][-1]
