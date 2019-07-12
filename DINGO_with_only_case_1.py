from copy import deepcopy
from plot_results import plot_results
from time import time
import torch
import torch.distributed as dist


def DINGO_with_only_case_1(model, worker, device, theta=1e-4,
                           max_iterations=100, max_communication_rounds=200,
                           gradient_norm_tolerance=1e-8, line_search_rho=1e-4, 
                           line_search_max_iterations=50):
    """Run the DINGO algorithm with only Case 1.

    Arguments:
        model (torch.nn.Module): A model to optimize.
        worker (Worker): A Worker class instance.
        device (torch.device): The device tensors will be allocated to.
        theta (float, optional): The hyperparameter theta in the DINGO
            algorithm.
        max_iterations (int, optional): The maximum number of iterations.
        max_communication_rounds (int, optional): The maximum number of
            communication rounds.
        gradient_norm_tolerance (float, optional): The smallest the norm of the
            full gradient can be before the algorithm stops.
        line_search_rho (float, optional): Armijo line search parameter.
        line_search_max_iterations (int, optional): The maximum number of line
            search iterations.
    """
    rank = dist.get_rank()
    dimension = sum([p.numel() for p in model.parameters()])
    num_workers = dist.get_world_size()-1
    cpu = torch.device("cpu")

    if rank == 0:
        print("\n{:-^116s}\n".format(" DINGO With Only Case 1 "))
        # Results are added to these lists and then plotted.
        cumulative_communication_rounds_list = [0]
        cumulative_time_list = [0]
        gradient_norm_list = []
        loss_list = []
        step_size_list = []
        test_accuracy_list = []
        # We will store a message about why the algorithm stopped.
        end_message = "max_iterations reached"

    iteration = 0
    total_communication_rounds = 0

    while iteration < max_iterations:
        if total_communication_rounds >= max_communication_rounds:
            end_message = 'max_communication_rounds reached'
            break

        #----------------------- DINGO Case 1 iteration -----------------------

        if rank == 0:
            # The driver will record how long each iteration takes.
            iteration_start_time = time()

        if iteration == 0:
            # This iteration requires an additional communication round.
            '''
            In subsequent iterations, the driver will broadcast the step-size
            to update workers' model.
            '''
            if rank > 0:
                local_loss = worker.get_local_loss(model).cpu()
                local_gradient = worker.get_local_gradient(model).cpu()
                local_loss_and_gradient = torch.cat([local_loss.reshape(1,1),
                                                     local_gradient], dim=0)
                # Workers send local objective value and gradient to driver.
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
                step_size = 0.0

        if rank == 0:
            # The driver broadcasts the step-size and gradient to all workers.
            # step_size is from previous line search.
            alpha_and_gradient = torch.cat(
                [torch.tensor(float(step_size), device=cpu).reshape(1,1),
                 gradient.cpu()], dim=0)
            dist.broadcast(alpha_and_gradient, 0)
            total_communication_rounds += 1
            gradient_norm = torch.norm(gradient)
            if gradient_norm <= gradient_norm_tolerance:
                end_message = 'gradient_norm_tolerance reached'
                break
            gradient_norm_squared = gradient_norm.pow(2)

        if rank > 0:
            alpha_and_gradient = torch.zeros(1+dimension, 1, device=cpu)
            dist.broadcast(alpha_and_gradient, 0)
            total_communication_rounds += 1
            alpha_and_gradient = alpha_and_gradient.to(device)
            step_size = alpha_and_gradient[0,0]
            gradient  = alpha_and_gradient[1:]
            if iteration > 0:
                # All workers update to current model.
                # This worker node has direction from previous line search.
                model = get_updated_model(model, direction, step_size)
            gradient_norm = torch.norm(gradient)
            if gradient_norm <= gradient_norm_tolerance:
                break

        # Update Direction.
        if rank == 0:
            direction, hessian_times_gradient = get_update_direction(model,
                worker, device, gradient)
            total_communication_rounds += 2
            # The Case 1 direction might not be a suitable descent direction.
            if (torch.mm(direction.transpose(0,1), hessian_times_gradient)
                > -theta*gradient_norm_squared):
                print("Warning: At iteration t={}, Case 1 direction does not"
                      "satisfy <p,Hg> <= -θ||g||".format(iteration))

        if rank > 0:
            direction = get_update_direction(model, worker, device, gradient)
            total_communication_rounds += 2

        # Line Search.
        if rank > 0:
            local_line_search_matrix = get_local_line_search_matrix(model,
                worker, direction, line_search_max_iterations)
            dist.reduce(local_line_search_matrix, 0)
            total_communication_rounds += 1

        if rank == 0:
            # The driver averages all workers' local line-search matrix.
            line_search_matrix = torch.zeros(1+dimension,
                line_search_max_iterations, device=cpu)
            dist.reduce(line_search_matrix, 0)
            total_communication_rounds += 1
            line_search_matrix = line_search_matrix.to(device)
            line_search_matrix *= 1.0/num_workers
            new_model, new_loss, new_gradient, line_search_exp, step_size \
                = line_search(model, line_search_matrix, gradient_norm_squared,
                              direction, hessian_times_gradient,
                              line_search_rho)
            iteration_time = time() - iteration_start_time

        #------------------------------ Printing ------------------------------
        # This time is not recorded.

        if rank == 0:
            loss_list.append(loss)
            gradient_norm_list.append(gradient_norm)
            step_size_list.append(step_size)
            # Recall that the driver stores the test dataset in its worker.
            test_accuracy = worker.get_accuracy(model)
            test_accuracy_list.append(test_accuracy)
            cumulative_communication_rounds_list.append(
                total_communication_rounds)
            cumulative_time_list.append(
                cumulative_time_list[-1] + iteration_time)
            print_row(iteration, total_communication_rounds, iteration_time,
                      1, line_search_exp, step_size, torch.norm(direction),
                      loss, gradient_norm, test_accuracy)

            loss = new_loss
            gradient = new_gradient
            model = new_model

        iteration += 1


    # Print final row.
    if rank == 0:
        loss_list.append(loss)
        gradient_norm = torch.norm(gradient)
        gradient_norm_list.append(gradient_norm)
        test_accuracy = worker.get_accuracy(model)
        test_accuracy_list.append(test_accuracy)

        print_row(iteration=iteration, objective_value=loss,
                  gradient_norm=gradient_norm,
                  test_accuracy=test_accuracy, is_final_row=True)
        
        print("\n{} after {:.2f} seconds\n".format(end_message,
              cumulative_time_list[-1]))

        plot_results(loss_list, gradient_norm_list, test_accuracy_list,
                     cumulative_communication_rounds_list, step_size_list,
                     label="DINGO", max_x=max_communication_rounds)


def print_row(iteration=0, cumulative_communication_rounds=0, iteration_time=0,
              case=1, line_search_exp=0, step_size=1, update_direction_norm=0,
              objective_value=0, gradient_norm=0, test_accuracy=0,
              is_final_row=False):
    """Print a row of the results table for DINGO.

    Arguments:
        iteration (int, optional): The iteration number.
        cumulative_communication_rounds (int, optional): The total number of
            communication rounds so far.
        iteration_time (float, optional): The runtime, in seconds, of the
            iteration.
        case (int, optional): The case the iteration belongs to.
        line_search_exp (int, optional): The line search exponent found.
        step_size (float, optional): The step size found.
        update_direction_norm (float, optional): The norm of the update
            direction.
        objective_value (float, optional): The objective value.
        gradient_norm (float, optional): The norm of the full gradient.
        test_accuracy (float, optional): The accuracy on the test dataset.
        is_final_row (bool, optional): Whether this is the final row of the
            results table.
    """
    header = ("{:^10s}{:^10s}{:^12s}{:^10s}{:^10s}{:^10s}{:^10s}"
              "{:^10s}{:^14s}{:^20s}").format("Iter (t)", "CCR", "Time (sec)",
                "Case", "LS Exp", "Step Size", "||p_t||", "f(w_t)",
                "||∇f(w_t)||", "Test Accuracy (%)")
    if (iteration)%20==0:
        # Print the header every 20 iterations.
        print(header)
    if is_final_row:
        prt = "{:^10d}{:62s}{:^10.2e}{:^14.2e}{:^20.2f}".format(
                iteration, '', objective_value, gradient_norm, test_accuracy)
        print(prt)
    else:
        prt = ("{:^10d}{:^10d}{:^12.2e}{:^10d}{:^10d}{:^10.2e}"
               "{:^10.2e}{:^10.2e}{:^14.2e}{:^20.2f}").format(
                iteration, cumulative_communication_rounds, iteration_time,
                case, line_search_exp, step_size, update_direction_norm, 
                objective_value, gradient_norm, test_accuracy)
        print(prt)


def get_update_direction(model, worker, device, gradient):
    """Returns the Case 1 update direction of DINGO. Also returns the
    Hessian-gradient product to the driver node.

    Arguments:
        model (torch.nn.Module): A model to optimize.
        worker (Worker): A Worker class instance.
        device (torch.device): The device tensors will be allocated to.
        gradient (torch.Tensor): The current full gradient.
    """
    rank = dist.get_rank()
    dimension = sum([p.numel() for p in model.parameters()])
    num_workers = dist.get_world_size()-1
    cpu = torch.device("cpu")

    if rank > 0:
        local_hessian_times_gradient \
            = worker.get_local_hessian_times_vector(model, gradient).cpu()
        local_direction = -1.0*worker.get_local_hessian_inverse_times_vector(
            model, gradient, "minresQLP").cpu()
        local_Hg_and_direction = torch.cat([local_hessian_times_gradient,
                                            local_direction], dim=1)
        dist.reduce(local_Hg_and_direction, 0)

    if rank == 0:
        Hg_and_direction = torch.zeros(dimension, 2, device=cpu)
        dist.reduce(Hg_and_direction, 0)
        Hg_and_direction = Hg_and_direction.to(device)
        Hg_and_direction *= 1.0/num_workers
        hessian_times_gradient = Hg_and_direction[:,0:1]
        direction = Hg_and_direction[:,1:2]
        dist.broadcast(direction.cpu(), 0)
        return direction, hessian_times_gradient

    if rank > 0:
        direction = torch.zeros(dimension, 1, device=cpu)
        dist.broadcast(direction, 0)
        direction = direction.to(device)
        return direction


###############################################################################
#--------------------------- Line Search Functions ---------------------------#


def get_updated_model(model, update_direction, step_size):
    """Returns a new model with parameters equal to the addition of
    step_size*update_direction and the input model parameters.

    Arguments:
        model (torch.nn.Module): A model.
        update_direction (torch.Tensor): The update direction.
        step_size (float): The step size.
    """
    new_model = deepcopy(model)
    with torch.no_grad():
        parameter_numels \
            = [parameter.numel() for parameter in new_model.parameters()]
        direction_split = torch.split(update_direction, parameter_numels)
        for i, parameter in enumerate(new_model.parameters(), 0):
            parameter += step_size*direction_split[i].reshape(parameter.shape)
    return new_model


def get_local_line_search_matrix(model, worker, update_direction,
                                 line_search_max_iterations):
    """Return a tensor where column k is the local objective value and local
    gradient at the point: weights + 0.5**k * update_direction.

    Args:
        model (torch.nn.Module): A model.
        worker (Worker): A Worker class instance.
        update_direction (torch.Tensor): The update direction.
        line_search_max_iterations (int): The maximum number of line search
            iterations.
    """
    local_vector_list = []
    line_search_exp = 0
    step_size = 1.0
    while line_search_exp < line_search_max_iterations:
        temp_model = get_updated_model(model, update_direction, step_size)
        temp_loss = worker.get_local_loss(temp_model).reshape(1,1).cpu()
        temp_gradient = worker.get_local_gradient(temp_model).cpu()
        temp_vector = torch.cat([temp_loss, temp_gradient], dim=0)
        local_vector_list.append(temp_vector)
        line_search_exp += 1
        step_size = step_size/2
    return torch.cat(local_vector_list, dim=1)


def line_search(model, line_search_matrix, gradient_norm_squared,
                update_direction, hessian_times_gradient, line_search_rho):
    """Compute the largest step-size that passes backtracking line search on
    the square of the norm of the gradient. Otherwise, return the smallest
    step-size.

    Args:
        model (torch.nn.Module): The current model.
        line_search_matrix (torch.Tensor): A matrix where column k is the
            objective value and gradient at the point:
            weights + 0.5**k * update_direction.
        gradient_norm_squared (float): The square of the norm of the full
            gradient.
        update_direction (torch.Tensor): The update direction.
        hessian_times_gradient (torch.Tensor): The current Hessian-gradient
            product.
        line_search_rho (float): Armijo line search parameter.
    """
    line_search_exp = 0
    step_size = 1.0
    line_search_max_iterations = line_search_matrix.shape[1]
    direction_dot_hessian_times_gradient = torch.mm(
        update_direction.transpose(0,1), hessian_times_gradient)
    new_loss = line_search_matrix[0,0]
    new_gradient = line_search_matrix[1:,0:1]
    while (torch.mm(new_gradient.transpose(0,1), new_gradient)
           > gradient_norm_squared
           + 2*step_size*line_search_rho*direction_dot_hessian_times_gradient
           and line_search_exp < line_search_max_iterations):
        line_search_exp += 1
        step_size = step_size/2
        new_loss = line_search_matrix[0,line_search_exp]
        new_gradient = line_search_matrix[1:,line_search_exp:line_search_exp+1]
    new_model = get_updated_model(model, update_direction, step_size)
    return new_model, new_loss, new_gradient, line_search_exp, step_size