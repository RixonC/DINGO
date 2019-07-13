from copy import deepcopy
from AIDE import AIDE
from DINGO import DINGO
from DINGO_with_only_case_1 import DINGO_with_only_case_1
from DiSCO import DiSCO
from GIANT import GIANT
from InexactDANE import InexactDANE
from SGD import Synchronous_SGD
from softmaxModel import softmaxModel
from worker import Worker
import torch
import torch.distributed as dist
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def main():
    """Compare the distributed optimization algorithms."""
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rank = dist.get_rank()
    number_of_workers = dist.get_world_size()-1

    #-------- Worker nodes load local trainset and driver loads testset -------

    data_dir = "./data/CIFAR10"
    trainset = datasets.CIFAR10(data_dir, train=True,
                                transform=transforms.ToTensor())
    if rank > 0:
        local_trainset_size = int(len(trainset)/number_of_workers)
        local_trainset = data.Subset(trainset,
                                     range(local_trainset_size * (rank-1),
                                           local_trainset_size * (rank)))
    else:
        # The driver node will record test accuracy.
        testset = datasets.CIFAR10(data_dir, train=False,
                                   transform=transforms.ToTensor())
        print("\nCIFAR10")
        print("trainset size = {:,}".format(len(trainset)))
        print("testset size = {:,}".format(len(testset)))

    #-------------------------- Load model to train ---------------------------

    model = softmaxModel(32*32*3, 10).to(device) # softmax model for CIFAR10
    if rank == 0:
        print("Softmax")
        print("dimension = {:,}".format(
            sum([p.numel() for p in model.parameters()])))

    #------------------ Choose criterion and regularization -------------------

    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    regularization_parameter = 1.0/len(trainset)

    #------------------------ Initialize worker class -------------------------

    '''
    A Worker class instance is held on each worker node. It contains the local
    trainset and local computation instructions, e.g., functions to compute the
    gradient on local data.

    A Worker class instance, containing the test dataset, is held on the driver
    node. This is used to compute the test accuracy.
    '''
    subproblem_maximum_iterations = 50
    subproblem_tolerance = 1e-4
    dataloader_processes = 0
    batch_size = 1000 # batch_size is the number of samples loaded at a time
    # when computing objective value, gradient, Hessian-vector products and
    # test accuracy.

    if rank > 0:
        worker = Worker(local_trainset, batch_size, dataloader_processes,
                        criterion, device, subproblem_tolerance,
                        subproblem_maximum_iterations,
                        regularization_parameter)
    else:
        worker = Worker(testset, batch_size, dataloader_processes,
                        criterion, device, subproblem_tolerance,
                        subproblem_maximum_iterations,
                        regularization_parameter)
        print("number of worker nodes = {:,}".format(number_of_workers))
        print("subproblem_tolerance = {:.2e}".format(subproblem_tolerance))
        print("subproblem_maximum_iterations = " +
              "{:,}".format(subproblem_maximum_iterations))

    #------------------------------ Train model -------------------------------

    max_communication_rounds = 500
    max_iterations = 2000

    DINGO(deepcopy(model), worker, device, theta=1e-4, phi=1e-6, 
          max_iterations = max_iterations,
          max_communication_rounds = max_communication_rounds)

#    DINGO_with_only_case_1(deepcopy(model), worker, device, theta=1e-4,
#                          max_iterations = max_iterations,
#                          max_communication_rounds = max_communication_rounds)

    GIANT(deepcopy(model), worker, device, max_iterations = max_iterations,
          max_communication_rounds = max_communication_rounds)

    DiSCO(deepcopy(model), worker, device, max_iterations = max_iterations,
          max_communication_rounds = max_communication_rounds,
          subproblem_tolerance = subproblem_tolerance,
          subproblem_maximum_iterations = subproblem_maximum_iterations)

    InexactDANE(deepcopy(model), worker, device, eta = 1.0, mu = 0.0, 
                subproblem_step_size = 1e-3, max_iterations = max_iterations,
                max_communication_rounds = max_communication_rounds)

    AIDE(deepcopy(model), worker, device, eta = 1.0, mu = 0.0, tau = 100,
         subproblem_step_size = 1e-3, max_iterations = max_iterations,
         max_communication_rounds = max_communication_rounds)

    Synchronous_SGD(deepcopy(model), worker, device, learning_rate = 1e-2,
                    max_iterations = max_iterations,
                    max_communication_rounds = max_communication_rounds)


if __name__ == "__main__":
    dist.init_process_group('mpi')
    main()
