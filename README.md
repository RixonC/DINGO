# DINGO: Distributed Newton-Type Method for Gradient-Norm Optimization

This code implements the distributed second-order optimization method DINGO from our arXiv paper [DINGO: Distributed Newton-Type Method for Gradient-Norm Optimization](https://arxiv.org/abs/1901.05134).
The implementation of the sub-problem solvers MINRES-QLP and LSMR uses modified code from:
1. Yang Liu and Fred Roosta. MINRES-QLP. GitHub repository, [https://github.com/syangliu/MINRES-QLP](https://github.com/syangliu/MINRES-QLP).
2. Dominique Orban, David Chin-lung Fong and Michael Saunders. pykrylov. GitHub repository, [https://github.com/PythonOptimizers/pykrylov](https://github.com/PythonOptimizers/pykrylov).


## Running Experiments

The code is built on-top of PyTorch and its distributed communication package `torch.distributed`.
Our code currently uses the MPI backend and can be modified to use other backends.
To install PyTorch with the MPI backend one must install PyTorch [from source](https://github.com/pytorch/pytorch#from-source).

Initializing experiments is done by running the `main.py` script.
The code implements and compares the distributed optimization methods DINGO, GIANT, DiSCO, InexactDANE, AIDE and synchronous stochastic gradient descent (Synchronous SGD).
Properties of the experiment, including hyper-parameters of the algorithms, can be changed in the `main.py` script.
For example, to run the code with 1 Driver and 2 Workers, one would use the command:
`mpirun -n 3 python main.py`.
Please use Python 3.

At the time of code upload, the `main.py` script uses the hyper-parameters used in Plot 1(a) of the paper.
Note that there might be slight variation in generated plots of InexactDANE, AIDE and Synchronous SGD to that of the plots in the paper.
This is because of the stochastic nature of Synchronous SGD and the sub-problem solver SVRG of InexactDANE.

## Models

This code trains models built from the widely used `torch.nn.Module` class. Therefore, many existing models can be easily imported into and trained by this code.

## Datasets

We efficiently handle datasets through the classes `torch.utils.data.Dataset` and `torch.utils.data.DataLoader`.
We currently use the `torchvision.datasets` to access datasets. 
Please change `data_dir`, in `main.py`, accordingly.

## Plots

All plots are saved in `./Plots`.

## Authors

1. Rixon Crane. School of Mathematics and Physics, University of Queensland, Australia. Email: r.crane(AT)uq.edu.au
2. Fred Roosta. School of Mathematics and Physics, University of Queensland, Australia, and International Computer Science Institute, Berkeley, USA. Email: fred.roosta(AT)uq.edu.au
