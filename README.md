# DINGO: Distributed Newton-Type Method for Gradient-Norm Optimization

This code implements the distributed second-order optimization method DINGO from our arXiv paper [DINGO: Distributed Newton-Type Method for Gradient-Norm Optimization](https://arxiv.org/abs/1901.05134).
The implementation of the sub-problem solvers MINRES-QLP and LSMR uses modified code from:
1. Yang Liu and Fred Roosta. MINRES-QLP. GitHub repository, [https://github.com/syangliu/MINRES-QLP](https://github.com/syangliu/MINRES-QLP).
2. Dominique Orban, David Chin-lung Fong and Michael Saunders. pykrylov. GitHub repository, [https://github.com/PythonOptimizers/pykrylov](https://github.com/PythonOptimizers/pykrylov).


## Running Experiments

The code is separated into two directories: `/MPI` and `/Serialized`.

### `/MPI`

The code under this directory implements DINGO on top of PyTorch with MPI support.
To install PyTorch with the MPI backend one must install PyTorch [from source](https://github.com/pytorch/pytorch#from-source).
Initializing experiments is done by running the `main.py` script.
Properties of the experiment, including hyper-parameters of DINGO, can be changed in the `main.py` script.
For example, to run DINGO with 1 Driver and 2 Workers, one would use the command:
`mpirun -n 3 python main.py`.

### `/Serialized`

This code runs the experiments in the arXiv paper.
It implements and compares the distributed optimization methods DINGO, GIANT, DiSCO, InexactDANE, AIDE, asynchronous stochastic gradient descent (Async-SGD) and synchronous stochastic gradient descent (Sync-SGD).
The code is designed to be easy to setup and run.
It simulates the distributed computing environment using class instances.
Therefore, it can be easily run on a desktop or laptop computer with minimal setup.
We recommend using the Anaconda Python distribution with Python 3.
The packages `pytorch` and `torchvision` can be easily installed using conda.
If you do not have a compatible Nvidia GPU or are having difficulty using CUDA, simply change `use_cuda` (on line 24 of `main.py` and `GMM_main.py`) to `False`.

Initializing experiments is done by running the `main.py` or `GMM_main.py` scripts.
Note that there might be slight variation in generated plots of InexactDANE, AIDE, Async-SGD and Sync-SGD to that of the plots in the paper.
This is because of the stochastic nature of Async-SGD, Sync-SGD and the sub-problem solver SVRG of InexactDANE.
1. `main.py`.
This runs experiments on the softmax regression and autoencoder problems.
Properties of the experiment, including hyper-parameters of the optimization methods, can be changed in the `main.py` script.
For example, the `main.py` script, at the time of code upload, runs the experiment in Plot 1(a) of the paper.
2. `GMM_main.py`.
This runs experiments on the Gaussian mixture model problem.
Properties of the experiment, including hyper-parameters of the optimization methods, can be changed in the `GMM_main.py`     script.
For example, the `GMM_main.py` script, at the time of code upload, runs the experiment in Plot 4(a) of the paper.


## Datasets

There is no need for any manual handling of datasets.
The Curves dataset comes pre-loaded.
The CIFAR10 and EMNIST Digits datasets will be automatically downloaded, by PyTorch, when needed.
Datasets are stored in `../Data`.


## Plots

All plots are saved in `./Plots`.

## Authors

1. Rixon Crane. School of Mathematics and Physics, University of Queensland, Australia. Email: r.crane(AT)uq.edu.au
2. Fred Roosta. School of Mathematics and Physics, University of Queensland, Australia, and International Computer Science Institute, Berkeley, USA. Email: fred.roosta(AT)uq.edu.au
