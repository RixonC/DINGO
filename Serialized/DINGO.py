# -*- coding: utf-8 -*-

from __future__ import division

from resultsRecorder import ResultsRecorder
from time import time
import torch


###############################################################################################
#---------------------------------------- H^\dagger g ----------------------------------------#

def get_hess_inv_grad(args):
    '''Worker i returns an approximate least-squares solution to H_i*x=g using MINRES-QLP.

    args = (worker, weights, gradient):
        worker: A worker class instance.
        weights: Current point.
        gradient: Full gradient at the current point.
    '''
    worker, weights, gradient = args
    return worker.hess_inv_vect(weights, gradient, method='MINRES-QLP')


def get_hess_inv_grad_list(workers, weights, gradient, config):
    '''Return a list [x_i], where x_i is worker i's approximate least-squares solution to H_i*x=g using MINRES-QLP.

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
        return [W.hess_inv_vect(weights, gradient, method='MINRES-QLP') for W in workers]


###############################################################################################
#-------------------------------- \tilde{H}^\dagger \tilde{g} --------------------------------#

def get_hess_tilde_inv_grad_tilde(args):
    '''Worker i returns an approximate least-squares solution to \tilde{H}_i*x=\tilde{g} using LSMR.

    args = (worker, weights, gradient):
        worker: A worker class instance.
        weights: Current point.
        gradient: Full gradient at the current point.
    '''
    worker, weights, gradient = args
    return worker.hess_tilde_inv_grad_tilde(weights, gradient)


def get_hess_tilde_inv_grad_tilde_list(workers, weights, gradient, config):
    '''Return a list [x_i], where x_i is worker i's approximate least-squares solution to \tilde{H}_i*x=\tilde{g} using LSMR.

    Args:
        workers: A list of worker class instances.
        weights: Current point.
        gradient: Full gradient at the current point.
        config: A dictionary of all necessary parameters.
    '''
    if config['use_multiprocessing']:
        args_list = [(w, weights, gradient) for w in workers]
        with torch.multiprocessing.Pool() as p:
            return p.map(get_hess_tilde_inv_grad_tilde, args_list)
    else:
        return [W.hess_tilde_inv_grad_tilde(weights, gradient) for W in workers]


###############################################################################################
#----------------------------------- Lagrangian Directions -----------------------------------#

def get_lagrangian_direction(args):
    '''Worker i returns an approximate solution to p_{t,i} in Case 3.

    args = (worker, weights, gradient):
        worker: A worker class instance.
        weights: Current point.
        gradient: Full gradient at the current point.
        hess_times_grad: Full Hessian-gradient product at the current point.
    '''
    worker, weights, gradient, hess_times_grad = args
    return worker.lagrangian_direction(weights, gradient, hess_times_grad)


def get_lagrangian_direction_list(workers, weights, gradient, hess_times_grad, config):
    '''Return a list [p_{t,1}]_{i=1}^{m}, where p_{t,i} is worker i's approximate solution to p_{t,i} in Case 3.

    Args:
        workers: A list of worker class instances.
        weights: Current point.
        gradient: Full gradient at the current point.
        hess_times_grad: Full Hessian-gradient product at the current point.
        config: A dictionary of all necessary parameters.
    '''
    if config['use_multiprocessing']:
        args_list = [(w, weights, gradient, hess_times_grad) for w in workers]
        with torch.multiprocessing.Pool() as p:
            return p.map(get_lagrangian_direction, args_list)
    else:
        return [W.lagrangian_direction(weights, gradient, hess_times_grad) for W in workers]


###############################################################################################
#------------------------------------------- DINGO -------------------------------------------#

def DINGO(workers, config, test_X=None, test_Y=None):
    '''Perform the DINGO algorithm.

    Args:
        workers: A list of worker class instances.
        config: A dictionary of all necessary parameters.
        test_X: test features.
        test_Y: test labels.
    '''
    print('\n ##########################################################################################################################')
    print('\n DINGO: theta = {}, phi = {} \n'.format(config['DINGO_theta'], config['DINGO_phi']))

    # if the subproblem fails then the algorithm stops
    subproblem_failed_end_message = 'CG failed when computing Lagrangian direction'

    # the Recorder class records, plots and saves results
    Recorder = ResultsRecorder('DINGO', config, test_X, test_Y)

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
        get_grad = lambda w: (1/config['num_partitions']) * sum([W.grad(w) for W in workers]) # [d,1]

        # results are added to these lists
        lists = {'cumulative_communication_rounds_list' : [],
                 'loss_list' : [],
                 'grad_norm_list' : [],
                 'avg_Lagrangian_lambda_list' : [],
                 'alpha_list' : [],
                 'ls_iters_list' : [],
                 'direction_norm_list' : [],
                 'inner_prod_list' : [],
                 'time_list' : [],
                 'cases_list' : [],
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

            if iteration == 0:
                # this iteration requires an additional two communication rounds
                gradient = get_grad(weights)
                total_communication_rounds += 2
                
            grad_norm = torch.norm(gradient)
            if grad_norm <= config['grad_tol']:
                end_message = 'grad_tol reached'
                break
            lists['grad_norm_list'].append(grad_norm)
            lists['loss_list'].append(get_loss())
            lists['test_accuracy_list'].append(Recorder.record_accuracy(weights)) # record test accuracy or error

            ###############################################################################################
            #------------------------------------- Update Direction --------------------------------------#

            hess_times_grad = (1/config['num_partitions']) * sum([W.hess_vect(weights, gradient) for W in workers])
            total_communication_rounds += 2

            hess_inv_grad_list = get_hess_inv_grad_list(workers, weights, gradient, config)
            p = (-1/config['num_partitions']) * sum([t[0] for t in hess_inv_grad_list]) # [d,1]

            if torch.mm(p.transpose(0,1), hess_times_grad) <= -config['DINGO_theta']*(grad_norm.pow(2)): # first try Case 1
                direction = p
                lists['cases_list'].append(1)
                lists['avg_Lagrangian_lambda_list'].append(0)
            else: # now try Case 2
                hess_tilde_inv_grad_tilde_list = get_hess_tilde_inv_grad_tilde_list(workers, weights, gradient, config)
                p = (-1/config['num_partitions']) * sum([t[0] for t in hess_tilde_inv_grad_tilde_list]) # [d,1]
                if torch.mm(p.transpose(0,1), hess_times_grad) <= -config['DINGO_theta']*(grad_norm.pow(2)):
                    direction = p
                    lists['cases_list'].append(2)
                    lists['avg_Lagrangian_lambda_list'].append(0)
                else: # Case 1 and Case 2 failed, so now try Case 3
                    total_communication_rounds += 2
                    lagrangian_directions_list = get_lagrangian_direction_list(workers, weights, gradient, hess_times_grad, config)

                    if None in lagrangian_directions_list: # CG failed to recover a solution
                        end_message = subproblem_failed_end_message
                        break

                    direction = (1/config['num_partitions']) * sum([t[0] for t in lagrangian_directions_list])
                    lagrangian_Lambdas = [t[1] for t in lagrangian_directions_list]
                    avgLagrangianLambda = sum(lagrangian_Lambdas)/len(lagrangian_Lambdas)
                    lists['avg_Lagrangian_lambda_list'].append(avgLagrangianLambda)
                    lists['cases_list'].append(3)

            direction_norm = torch.norm(direction)
            lists['direction_norm_list'].append(direction_norm)
            lists['inner_prod_list'].append(torch.mm(direction.transpose(0,1), hess_times_grad)/(direction_norm*torch.norm(hess_times_grad)))

            ###############################################################################################
            #---------------------------------------- Line Search ----------------------------------------#
            ls_iters = 0
            alpha = config['line_search_start_val']
            new_weights = weights + alpha * direction
            grad_new_weights = get_grad(new_weights)
            total_communication_rounds += 2
            direction_dot_hess_times_grad = torch.mm(direction.transpose(0,1), hess_times_grad)
            grad_norm2 = grad_norm.pow(2)

            while torch.mm(grad_new_weights.transpose(0,1), grad_new_weights) \
                    > grad_norm2 + 2*alpha*config['line_search_rho']*direction_dot_hess_times_grad \
                    and ls_iters < config['line_search_max_iterations']:

                alpha = alpha/2
                new_weights = weights + alpha * direction
                grad_new_weights = get_grad(new_weights)
                ls_iters += 1

            weights = new_weights
            gradient = grad_new_weights
            lists['alpha_list'].append(alpha)
            lists['ls_iters_list'].append(ls_iters)
            lists['cumulative_communication_rounds_list'].append(total_communication_rounds)
            lists['time_list'].append(time() - start_time)
            Recorder.print_row(iteration, lists) # print the current row of the results table
            iteration += 1

        if end_message != subproblem_failed_end_message:
            lists['loss_list'].append(get_loss())
            lists['grad_norm_list'].append(torch.norm(get_grad(weights)))
            lists['test_accuracy_list'].append(Recorder.record_accuracy(weights))
        
        Recorder.print_row(iteration, lists, final_row=True) # print the last row of the results table

        # print why the algorithm stopped
        print('\n     ' + end_message + ' in {:.2f} seconds'.format(sum(lists['time_list'])))

        if config['save_plot_data']:
            Recorder.save_plots_data(weights, end_message, lists)
        
        Recorder.print_plots(end_message, lists, subproblem_failed_end_message) # generate and save plots

        # return the final: weights/point, loss, norm of gradient, and test accuracy or error
        return weights, lists['loss_list'][-1], lists['grad_norm_list'][-1], lists['test_accuracy_list'][-1]
