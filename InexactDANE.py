from plot_results import plot_results
from time import time
import torch
import torch.distributed as dist


def InexactDANE(model, worker, device, eta=1.0, mu=0.0, 
                subproblem_step_size=1.0, max_iterations=100, 
                max_communication_rounds=200, gradient_norm_tolerance=1e-8):
    """Run the InexactDANE algorithm.

    Arguments:
        model (torch.nn.Module): A model to optimize.
        worker (Worker): A Worker class instance.
        device (torch.device): The device tensors will be allocated to.
        eta (float, optional): The hyperparameter eta in the InexactDANE 
            algorithm.
        mu (float, optional): The hyperparameter mu in the InexactDANE 
            algorithm.
        subproblem_step_size (float, optional): The step size used by the 
            subproblem solver.
        max_iterations (int, optional): The maximum number of iterations.
        max_communication_rounds (int, optional): The maximum number of 
            communication rounds.
        gradient_norm_tolerance (float, optional): The smallest the norm of the
            full gradient can be before the algorithm stops.
    """
    rank = dist.get_rank()
    dimension = sum([p.numel() for p in model.parameters()])
    num_workers = dist.get_world_size()-1
    cpu = torch.device("cpu")

    if rank == 0:
        print("\n{:-^76s}\n".format(" InexactDANE "))
        # Results are added to these lists and then plotted.
        cumulative_communication_rounds_list = [0]
        cumulative_time_list = [0]
        gradient_norm_list = []
        loss_list = []
        test_accuracy_list = []
        # We will store a message about why the algorithm stopped.
        end_message = "max_iterations reached"

    iteration = 0
    total_communication_rounds = 0

    while iteration < max_iterations:
        if total_communication_rounds >= max_communication_rounds:
            end_message = 'max_communication_rounds reached'
            break
        
        #----------------------- InexactDANE iteration ------------------------

        if rank == 0:
            # The driver will record how long each iteration takes.
            iteration_start_time = time()

        if iteration > 0:
            # This iteration requires an initial update to workers' model.
            if rank == 0:
                # new_weights is from previous iteration.
                model = replace_model_weights(model, new_weights)
                dist.broadcast(new_weights.cpu(), 0)
                total_communication_rounds += 1

            if rank > 0:
                new_weights = torch.zeros(dimension, 1, device=cpu)
                dist.broadcast(new_weights, 0)
                total_communication_rounds += 1
                new_weights = new_weights.to(device)
                model = replace_model_weights(model, new_weights)

        if rank > 0:
            local_loss = worker.get_local_loss(model).cpu()
            local_gradient = worker.get_local_gradient(model).cpu()
            local_loss_and_gradient = torch.cat([local_loss.reshape(1,1),
                                                 local_gradient], dim=0)
            # All workers send local objective value and gradient to driver.
            dist.reduce(local_loss_and_gradient, 0)
            total_communication_rounds += 1

        if rank == 0:
            loss_and_gradient = torch.zeros(1+dimension, 1, device=cpu)
            dist.reduce(loss_and_gradient, 0)
            total_communication_rounds += 1
            loss_and_gradient = loss_and_gradient.to(device)
            loss_and_gradient *= 1.0/num_workers
            loss = loss_and_gradient[0,0]
            gradient = loss_and_gradient[1:]
            dist.broadcast(gradient.cpu(), 0)
            total_communication_rounds += 1
            gradient_norm = torch.norm(gradient)
            if gradient_norm <= gradient_norm_tolerance:
                end_message = 'gradient_norm_tolerance reached'
                break

        if rank > 0:
            gradient = torch.zeros(dimension, 1, device=cpu)
            dist.broadcast(gradient, 0)
            total_communication_rounds += 1
            gradient = gradient.to(device)
            gradient_norm = torch.norm(gradient)
            if gradient_norm <= gradient_norm_tolerance:
                break
            local_solution = worker.get_InexactDANE_subproblem_solution(
                model, gradient, subproblem_step_size, eta, mu).cpu()
            dist.reduce(local_solution, 0)
            total_communication_rounds += 1
        
        if rank == 0:
            new_weights = torch.zeros(dimension, 1, device=cpu)
            dist.reduce(new_weights, 0)
            total_communication_rounds += 1
            new_weights = new_weights.to(device)
            new_weights *= 1.0/num_workers
            iteration_time = time() - iteration_start_time
            
        #------------------------------ Printing ------------------------------
        # This time is not recorded.

        if rank == 0:
            loss_list.append(loss)
            gradient_norm_list.append(gradient_norm)
            # Recall that the driver stores the test dataset in its worker.
            test_accuracy = worker.get_accuracy(model)
            test_accuracy_list.append(test_accuracy)
            cumulative_communication_rounds_list.append(
                total_communication_rounds)
            cumulative_time_list.append(
                cumulative_time_list[-1] + iteration_time)
            print_row(iteration, total_communication_rounds, iteration_time,
                      loss, gradient_norm, test_accuracy)

        iteration += 1
    
    
    # Print final row.
    if (iteration == max_iterations
        or total_communication_rounds >= max_communication_rounds):
        # Need to first get objective value and gradient norm on final model.
        if rank == 0:
            model = replace_model_weights(model, new_weights)
            dist.broadcast(new_weights.cpu(), 0)
            total_communication_rounds += 1

        if rank > 0:
            new_weights = torch.zeros(dimension, 1, device=cpu)
            dist.broadcast(new_weights, 0)
            total_communication_rounds += 1
            new_weights = new_weights.to(device)
            model = replace_model_weights(model, new_weights)
            local_loss = worker.get_local_loss(model).cpu()
            local_gradient = worker.get_local_gradient(model).cpu()
            local_loss_and_gradient = torch.cat([local_loss.reshape(1,1),
                                                 local_gradient], dim=0)
            dist.reduce(local_loss_and_gradient, 0)

        if rank == 0:
            loss_and_gradient = torch.zeros(1+dimension, 1, device=cpu)
            dist.reduce(loss_and_gradient, 0)
            loss_and_gradient = loss_and_gradient.to(device)
            loss_and_gradient *= 1.0/num_workers
            loss = loss_and_gradient[0,0]
            gradient = loss_and_gradient[1:]
            gradient_norm = torch.norm(gradient)

    if rank == 0:
        loss_list.append(loss)
        gradient_norm_list.append(gradient_norm)
        test_accuracy = worker.get_accuracy(model)
        test_accuracy_list.append(test_accuracy)

        print_row(iteration=iteration, objective_value=loss,
                  gradient_norm=gradient_norm, test_accuracy=test_accuracy, 
                  is_final_row=True)
        
        print("\n{} after {:.2f} seconds\n".format(end_message,
              cumulative_time_list[-1]))

        plot_results(loss_list, gradient_norm_list, test_accuracy_list,
                     cumulative_communication_rounds_list, label="InexactDANE",
                     max_x=max_communication_rounds)


def print_row(iteration=0, cumulative_communication_rounds=0, iteration_time=0,
              objective_value=0, gradient_norm=0, test_accuracy=0, 
              is_final_row=False):
    """Print a row of the results table for InexactDANE.

    Arguments:
        iteration (int, optional): The iteration number.
        cumulative_communication_rounds (int, optional): The total number of
            communication rounds so far.
        iteration_time (float, optional): The runtime, in seconds, of the
            iteration.
        objective_value (float, optional): The objective value.
        gradient_norm (float, optional): The norm of the full gradient.
        test_accuracy (float, optional): The accuracy on the test dataset.
        is_final_row (bool, optional): Whether this is the final row of the
            results table.
    """
    header = "{:^10s}{:^10s}{:^12s}{:^10s}{:^14s}{:^20s}".format(
        "Iter (t)", "CCR", "Time (sec)", "f(w_t)", "||∇f(w_t)||", 
        "Test Accuracy (%)")
    if (iteration)%20==0:
        # Print the header every 20 iterations.
        print(header)
    if is_final_row:
        prt = "{:^10d}{:22s}{:^10.2e}{:^14.2e}{:^20.2f}".format(
                iteration, '', objective_value, gradient_norm, test_accuracy)
        print(prt)
    else:
        prt = ("{:^10d}{:^10d}{:^12.2e}{:^10.2e}{:^14.2e}"
               "{:^20.2f}").format(iteration, cumulative_communication_rounds, 
                iteration_time, objective_value, 
                gradient_norm, test_accuracy)
        print(prt)


def replace_model_weights(model, new_weights):
    """Replace the model parameters with new parameters.

    Arguments:
        model (torch.nn.Module): A model.
        new_weights (torch.Tensor): A column tensor of new parameters.
    """
    with torch.no_grad():
        parameter_numels \
            = [parameter.numel() for parameter in model.parameters()]
        weights_split = torch.split(new_weights, parameter_numels)
        for i, parameter in enumerate(model.parameters(), 0):
            parameter.zero_()
            parameter += weights_split[i].reshape(parameter.shape)
    return model