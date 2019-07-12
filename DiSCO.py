from copy import deepcopy
from plot_results import plot_results
from time import time
import torch
import torch.distributed as dist


def DiSCO(model, worker, device, max_iterations=100,
          max_communication_rounds=200, gradient_norm_tolerance=1e-8, 
          subproblem_tolerance=1e-4, subproblem_maximum_iterations=50):
    """Run the DiSCO algorithm.

    Arguments:
        model (torch.nn.Module): A model to optimize.
        worker (Worker): A Worker class instance. The worker on the driver is
            used to record test accuracy.
        device (torch.device): The device tensors will be allocated to.
        max_iterations (int, optional): The maximum number of iterations.
        max_communication_rounds (int, optional): The maximum number of
            communication rounds.
        gradient_norm_tolerance (float, optional): The smallest the norm of the
            full gradient can be before the algorithm stops.
        subproblem_tolerance (float, optional): The tolerance used by the
            subproblem solvers.
        subproblem_maximum_iterations (int, optional): The maximum number of
            iterations used by the subproblem solver.
    """
    rank = dist.get_rank()
    dimension = sum([p.numel() for p in model.parameters()])
    num_workers = dist.get_world_size()-1
    cpu = torch.device("cpu")

    if rank == 0:
        print("\n{:-^86s}\n".format(" DiSCO "))
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
    subproblem_failed = False

    while iteration < max_iterations:
        if total_communication_rounds >= max_communication_rounds:
            end_message = 'max_communication_rounds reached'
            break

        #-------------------------- DiSCO iteration ---------------------------

        if rank == 0:
            # The driver will record how long each iteration takes.
            iteration_start_time = time()

        if iteration > 0:
            # This iteration requires an initial update to workers' model.
            if rank == 0:
                # direction is from previous iteration.
                dist.broadcast(direction.cpu(), 0)
                total_communication_rounds += 1

            if rank > 0:
                direction = torch.zeros(dimension, 1, device=cpu)
                dist.broadcast(direction, 0)
                total_communication_rounds += 1
                direction = direction.to(device)
                model = get_updated_model(model, direction)

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

        PCG_result = distributed_PCG_algorithm(model, worker, device, gradient,
            subproblem_tolerance, subproblem_maximum_iterations)

        if rank > 0:
            if PCG_result is None: # PCG failed
                break
            total_communication_rounds += PCG_result

        if rank == 0:
            if PCG_result is None: # PCG failed
                end_message = 'PCG failed'
                subproblem_failed = True
                break
            v, delta, PCG_communication_rounds = PCG_result
            total_communication_rounds += PCG_communication_rounds
            direction = (-1.0 / (1 + delta)) * v
            new_model = get_updated_model(model, direction)
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
                      torch.norm(direction), loss, gradient_norm,
                      test_accuracy)

            model = new_model

        iteration += 1


    # Print final row.
    if (iteration == max_iterations
        or total_communication_rounds >= max_communication_rounds):
        # Need to first get objective value and gradient norm on final model.
        if rank == 0:
            dist.broadcast(direction.cpu(), 0)

        if rank > 0:
            direction = torch.zeros(dimension, 1, device=cpu)
            dist.broadcast(direction, 0)
            direction = direction.to(device)
            model = get_updated_model(model, direction)
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
                     cumulative_communication_rounds_list, label="DiSCO",
                     max_x=max_communication_rounds, failed=subproblem_failed)


def print_row(iteration=0, cumulative_communication_rounds=0, iteration_time=0,
              update_direction_norm=0, objective_value=0, gradient_norm=0,
              test_accuracy=0, is_final_row=False):
    """Print a row of the results table for DiSCO.

    Arguments:
        iteration (int, optional): The iteration number.
        cumulative_communication_rounds (int, optional): The total number of
            communication rounds so far.
        iteration_time (float, optional): The runtime, in seconds, of the
            iteration.
        update_direction_norm (float, optional): The norm of the update
            direction.
        objective_value (float, optional): The objective value.
        gradient_norm (float, optional): The norm of the full gradient.
        test_accuracy (float, optional): The accuracy on the test dataset.
        is_final_row (bool, optional): Whether this is the final row of the
            results table.
    """
    header = "{:^10s}{:^10s}{:^12s}{:^10s}{:^10s}{:^14s}{:^20s}".format(
        "Iter (t)", "CCR", "Time (sec)", "||p_t||", "f(w_t)", "||∇f(w_t)||", 
        "Test Accuracy (%)")
    if (iteration)%20==0:
        # Print the header every 20 iterations.
        print(header)
    if is_final_row:
        prt = "{:^10d}{:32s}{:^10.2e}{:^14.2e}{:^20.2f}".format(
                iteration, '', objective_value, gradient_norm, test_accuracy)
        print(prt)
    else:
        prt = ("{:^10d}{:^10d}{:^12.2e}{:^10.2e}{:^10.2e}{:^14.2e}"
               "{:^20.2f}").format(iteration, cumulative_communication_rounds, 
                iteration_time, update_direction_norm, objective_value, 
                gradient_norm, test_accuracy)
        print(prt)


def get_updated_model(model, update_direction):
    """Returns a new model with parameters equal to the addition of
    update_direction and the input model parameters.

    Arguments:
        model (torch.nn.Module): A model.
        update_direction (torch.Tensor): The update direction.
    """
    new_model = deepcopy(model)
    with torch.no_grad():
        parameter_numels \
            = [parameter.numel() for parameter in new_model.parameters()]
        direction_split = torch.split(update_direction, parameter_numels)
        for i, parameter in enumerate(new_model.parameters(), 0):
            parameter += direction_split[i].reshape(parameter.shape)
    return new_model


def distributed_PCG_algorithm(model, worker, device, gradient,
                              relative_residual_tolerance, max_iterations):
    """Run the distributed PCG algorithm. This implimentation does not use
    preconditioning.

    Arguments:
        model (torch.nn.Module): A model to optimize.
        worker (Worker): A Worker class instance.
        device (torch.device): The device tensors will be allocated to.
        gradient (torch.Tensor): The current full gradient.
        relative_residual_tolerance (float): The relative residual
            tolerance.
        max_iterations (int): The maximum number of iterations.
    """
    rank = dist.get_rank()
    dimension = sum([p.numel() for p in model.parameters()])
    cpu = torch.device("cpu")

    if rank == 0:
        num_workers = dist.get_world_size()-1
        gradient_norm = torch.norm(gradient)
        # Initialization.
        v_t = torch.zeros(dimension, 1, device=device)
        r_t = gradient.clone()
        s_t = r_t.clone()
        u_t = s_t.clone()
        hessian_times_v_t = torch.zeros(dimension, 1, device=device)

    # Repeat.
    communication_rounds = 0
    for iteration in range(max_iterations):
        if rank == 0:
            # The 0 in temp indicates that PCG has not yet failed.
            temp = torch.cat([torch.zeros((1,1), device=cpu), u_t.cpu()],
                             dim=0)
            dist.broadcast(temp, 0)
            communication_rounds += 1

        if rank > 0:
            temp = torch.zeros(1+dimension, 1, device=cpu)
            dist.broadcast(temp, 0)
            communication_rounds += 1
            if int(temp[0,0]) == 1: # PCG failed in previous iteration
                return None
            if int(temp[0,0]) == -1: # Relative residual tolerance reached
                break
            u_t = temp[1:].to(device)
            local_hessian_times_u_t = worker.get_local_hessian_times_vector(
                model, u_t).cpu()
            dist.reduce(local_hessian_times_u_t, 0)
            communication_rounds += 1

        if rank == 0:
            hessian_times_u_t = torch.zeros(dimension, 1, device=cpu)
            dist.reduce(hessian_times_u_t, 0)
            communication_rounds += 1
            hessian_times_u_t = hessian_times_u_t.to(device)
            hessian_times_u_t *= 1.0/num_workers
            uHu = torch.mm(u_t.transpose(0,1), hessian_times_u_t)
            if uHu <= 0: # PCG has failed
                temp = torch.cat([torch.ones((1,1), device=cpu),
                                torch.zeros((dimension,1), device=cpu)], dim=0)
                # The 1 indicates that PCG failed.
                dist.broadcast(temp, 0)
                return None
            alpha_t = torch.mm(r_t.transpose(0,1), s_t) / uHu
            v_t_plus_1 = v_t + alpha_t*u_t
            hessian_times_v_t_plus_1 \
                = hessian_times_v_t + alpha_t*hessian_times_u_t
            r_t_plus_1 = r_t - alpha_t*hessian_times_u_t
            s_t_plus_1 = r_t_plus_1.clone()
            beta_t = torch.mm(r_t_plus_1.transpose(0,1),
                              s_t_plus_1) / torch.mm(r_t.transpose(0,1), s_t)
            u_t_plus_1 = s_t_plus_1 + beta_t*u_t

            delta = torch.sqrt(torch.mm(v_t_plus_1.transpose(0,1),
                                        hessian_times_v_t)
                               + alpha_t*torch.mm(v_t_plus_1.transpose(0,1),
                                                  hessian_times_u_t))

            v_t = v_t_plus_1.clone()
            r_t = r_t_plus_1.clone()
            s_t = s_t_plus_1.clone()
            u_t = u_t_plus_1.clone()
            hessian_times_v_t = hessian_times_v_t_plus_1.clone()

            if (torch.norm(hessian_times_v_t - gradient) / gradient_norm
                <= relative_residual_tolerance):
                temp = torch.cat([-torch.ones((1,1), device=cpu),
                                torch.zeros((dimension,1), device=cpu)], dim=0)
                # The -1 indicates to break.
                dist.broadcast(temp, 0)
                communication_rounds += 1
                break

    if rank == 0:
        return v_t, delta, communication_rounds
    else:
        return communication_rounds
