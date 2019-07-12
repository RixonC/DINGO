import matplotlib.pyplot as plt


def plot_results(loss_list, grad_norm_list, test_accuracy_list, x_list, 
                 step_size_list=[], label="DINGO", max_x=float('inf'), 
                 xlabel="Communication Rounds", failed=False):
    """Plot the results of an algorithm.

    Arguments:
        loss_list (list[float]): A list of the objective value at each 
            iteration.
        grad_norm_list (list[float]): A list of the norm of the gradient at 
            each iteration.
        test_accuracy_list (list[float]): A list of the accuracy on the test
            dataset at each iteration.
        x_list (list[int]): A list of horizontal axis values.
        step_size_list (list[float], optional): A list of the step size used 
            at each iteration.
        label (str, optional): The label in the plots.
        max_x (float, optional): The maximum value on the horizontal axes.
        xlabel (str, optional): The label on the horizontal axes.
        failed (bool, optional): Whether the algorithm failed.
    """
    if label == 'AIDE':
        colour = 'r'
        style = ':'
    elif label == 'Asynchronous SGD':
        colour = 'y'
        style = '--'
    elif label == 'DINGO':
        colour = 'k'
        style = '-'
    elif label == 'DiSCO':
        colour = 'c'
        style = '-.'
    elif label == "GIANT":
        colour = 'g'
        style  = '-.'
    elif label == 'InexactDANE':
        colour = 'm'
        style = ':'
    else:
        colour = 'b'
        style = '--'
    
    plt.figure(1, figsize=(20,4))
    plt.subplots_adjust(wspace = 0.3)
    
    plt.subplot(141)
    plt.plot(x_list, loss_list, color=colour, linestyle=style, label=label)
    if failed:
        plt.plot(x_list[-1:], loss_list[-1:], color=colour, linestyle=style, 
                 label=label, marker='x')
    plt.xlabel(xlabel)
    plt.ylabel(r'Objective Function: $f\:(\mathbf{w})$')
    if x_list[-1] > max_x:
        plt.xlim(right = max_x)
    
    plt.subplot(142)
    plt.semilogy(x_list, grad_norm_list, color=colour, linestyle=style, 
                 label=label)
    if failed:
        plt.semilogy(x_list[-1:], grad_norm_list[-1:], color=colour, 
                     linestyle=style, label=label, marker='x')
    plt.xlabel(xlabel)
    plt.ylabel(r'Gradient Norm: $||\nabla f(\mathbf{w})||$')
    if x_list[-1] > max_x:
        plt.xlim(right = max_x)
    
    plt.subplot(143)
    plt.plot(x_list, test_accuracy_list, color=colour, linestyle=style, 
             label=label)
    if failed:
        plt.plot(x_list[-1:], test_accuracy_list[-1:], color=colour, 
                 linestyle=style, label=label, marker='x')
    plt.xlabel(xlabel)
    plt.ylabel('Test Classification Accuracy (%)')
    if x_list[-1] > max_x:
        plt.xlim(right = max_x)
    
    xlabel = 'Iteration'
    
    if len(step_size_list) > 0:
        plt.subplot(144)
        plt.semilogy(step_size_list, color=colour, linestyle=style, 
                     label=label)
        if failed:
            plt.semilogy([len(step_size_list)-1], step_size_list[-1:], 
                         color=colour, linestyle=style, label=label, 
                         marker='x')
        plt.xlabel(xlabel)
        plt.ylabel(r'Line Search: $\alpha$')
    
    plt.savefig("./plots/plot.pdf", bbox_inches='tight')