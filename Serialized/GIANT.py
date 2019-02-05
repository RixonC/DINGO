# -*- coding: utf-8 -*-

from __future__ import division

from resultsRecorder import ResultsRecorder
from time import time
import torch


###############################################################################################
#------------------------------------------ H^{-1} g -----------------------------------------#

def get_hess_inv_grad(args):
    '''Worker i returns an approximate solution to H_i*x=g using CG.

    args = (worker, weights, gradient):
        worker: A worker class instance.
        weights: Current point.
        gradient: Full gradient at the current point.
    '''
    worker, weights, gradient = args
    return worker.hess_inv_vect(weights, gradient, method='CG')


def get_hess_inv_grad_list(workers, weights, gradient, config):
    '''Return a list [x_i], where x_i is worker i's approximate solution to H_i*x=g using CG.

    Args:
        workers: A list of worker class instances.
        weights: Current point.
        gradient: Full gradient at the current point.
        config: A dictionary of all necessary parameters.
    '''
    if config['use_multiprocessing']:
        args_list = [(w, weights, gradient) for w in workers]
        with torch.multiprocessing.Pool() as p:
            return p.map(get_hess_inv_grad, args_list)
    else:
        return [W.hess_inv_vect(weights, gradient, method='CG') for W in workers]


###############################################################################################
#------------------------------------------- GIANT -------------------------------------------#

def GIANT(workers, config, test_X=None, test_Y=None):
    '''Perform the GIANT algorithm.

    Args:
        workers: A list of worker class instances.
        config: A dictionary of all necessary parameters.
        test_X: test features.
        test_Y: test labels.
    '''
    print('\n ###################################################################################################################')
    print('\n GIANT \n')

    # if the subproblem fails then the algorithm stops
    subproblem_failed_end_message = 'pAp<=0 in CG'

    # the Recorder class records, plots and saves results
    Recorder = ResultsRecorder('GIANT', config, test_X, test_Y)

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
        get_loss = lambda w: (1/config['num_partitions']) * sum([W.loss(w) for W in workers]) # []
        get_grad = lambda: (1/config['num_partitions']) * sum([W.grad(weights) for W in workers]) # [d,1]

        # results are added to these lists
        lists = {'loss_list' : [],
                 'grad_norm_list' : [],
                 'iter_solver_list' : [],
                 'alpha_list' : [],
                 'ls_iters_list' : [],
                 'direction_norm_list' : [],
                 'inner_prod_list' : [],
                 'time_list' : [],
                 'cumulative_communication_rounds_list' : [],
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

            gradient = get_grad() # [d,1]
            grad_norm = torch.norm(gradient) # []
            if grad_norm <= config['grad_tol']:
                end_message = 'grad_tol reached'
                break
            total_communication_rounds += 2

            hess_inv_grad_list = get_hess_inv_grad_list(workers, weights, gradient, config)
            total_communication_rounds += 2

            if None in hess_inv_grad_list: # CG failed to recover a solution
                end_message = subproblem_failed_end_message
                break

            lists['grad_norm_list'].append(grad_norm)
            loss = get_loss(weights)
            lists['loss_list'].append(loss)
            lists['test_accuracy_list'].append(Recorder.record_accuracy(weights)) # record test accuracy or error

            direction = (-1/config['num_partitions']) * sum([t[0] for t in hess_inv_grad_list]) # [d,1]
            direction_norm = torch.norm(direction)
            lists['direction_norm_list'].append(direction_norm)
            solver_iters_list = [t[2] for t in hess_inv_grad_list]
            average_solver_iters = sum(solver_iters_list)/len(solver_iters_list)
            lists['iter_solver_list'].append(average_solver_iters)

            ###############################################################################################
            #---------------------------------------- Line Search ----------------------------------------#
            ls_iters = 0
            alpha = config['line_search_start_val']
            new_weights = weights + alpha * direction # [d,1]
            loss_new_weights = get_loss(new_weights) # []
            direction_dot_grad = torch.mm(direction.transpose(0,1), gradient) # [1]
            lists['inner_prod_list'].append(direction_dot_grad/(grad_norm*direction_norm))

            while loss_new_weights > loss + alpha * config['line_search_rho'] * direction_dot_grad \
                    and ls_iters < config['line_search_max_iterations']:

                alpha = alpha/2
                new_weights = weights + alpha * direction # [d,1]
                loss_new_weights = get_loss(new_weights) # []
                ls_iters += 1

            weights = new_weights # [d,1]
            lists['alpha_list'].append(alpha)
            lists['ls_iters_list'].append(ls_iters)
            total_communication_rounds += 2
            lists['cumulative_communication_rounds_list'].append(total_communication_rounds)
            lists['time_list'].append(time() - start_time)
            Recorder.print_row(iteration, lists) # print the current row of the results table
            iteration += 1

        lists['loss_list'].append(get_loss(weights))
        lists['grad_norm_list'].append(torch.norm(get_grad()))
        lists['test_accuracy_list'].append(Recorder.record_accuracy(weights))
        Recorder.print_row(iteration, lists, final_row=True) # print the last row of the results table

        # print why the algorithm stopped
        print('\n     ' + end_message + ' in {:.2f} seconds'.format(sum(lists['time_list'])))

        if config['save_plot_data']:
            Recorder.save_plots_data(weights, end_message, lists)
            
        Recorder.print_plots(end_message, lists, subproblem_failed_end_message) # generate and save plots

        # return the final: weights/point, loss, norm of gradient, and test accuracy or error
        return weights, lists['loss_list'][-1], lists['grad_norm_list'][-1], lists['test_accuracy_list'][-1]
