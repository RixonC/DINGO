# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from scipy import sparse
from scipy.io import loadmat
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class myDataLoader(object):
    '''This class loads and processes data.

    Args:
        dataset: The name of the dataset to load and process.
        data_directory: The directory containing the dataset.
    '''
    def __init__(self, dataset, data_directory):
        self.dataset = dataset
        self.directory = data_directory


    def process_data(self, X, Y, idx, objective_function, autoencoder_layers=None, is_test_data=False):
        '''Process data for the given objective function.'''
        n, p= X.shape
        if objective_function == 'Autoencoder':
            # Y is unused for Autoencoder.
            # the total number of weights in the autoencoder is:
            d = sum([autoencoder_layers[k+1]*(autoencoder_layers[k]+1) for k in range(len(autoencoder_layers)-1)])
        if objective_function == 'softmax':
            Classes = sorted(set(Y.numpy()))
            Total_C = len(Classes)
            d = p*(Total_C-1)
            I = np.ones(n)
            X_label = np.array([i for i in range(n)])
            Y = sparse.coo_matrix((I,(X_label, Y.numpy())), shape=(n, Total_C))
            Y = Y.tocsr().toarray()
            if is_test_data:
                Y = torch.Tensor(Y) # [n,C]
            else:
                Y = Y[:,:-1]
                Y = torch.Tensor(Y) # [n,C-1]
        return X, Y, d


    def load_train_data(self):
        '''Return the training data.'''
        if self.dataset == 'CIFAR10':
            data = datasets.CIFAR10(self.directory, train=True, download=True, transform=transforms.ToTensor())
            idx = 5
        if self.dataset == 'Curves':
            # n = 20,000 and p = 784
            data = loadmat(self.directory + "/curves.mat")
            X = torch.Tensor(data['X']).transpose(0,1)
            Y = torch.Tensor(data['y']).transpose(0,1).reshape(-1)
            idx = 0 # we only use Curves for Autoencoder
            return X, Y, idx
        if self.dataset == 'EMNIST_digits':
            # n = 240,000 and p = 784
            data = datasets.EMNIST(self.directory, 'digits', train=True, download=True, transform=transforms.ToTensor())
            idx = 5
        # the following converts it to a [n,-1] tensor
        n = len(data)
        data = torch.utils.data.DataLoader(data, batch_size=n) # one batch contains all the data
        X, Y = iter(data).next() # extract the one batch
        # X is in the form [n, num_channels, ?, ?]
        X = X.reshape(n,-1) # we leave Y as an array
        return X, Y, idx


    def load_test_data(self):
        '''Return the test data.'''
        if self.dataset == 'CIFAR10':
            data = datasets.CIFAR10(self.directory, train=False, download=True, transform=transforms.ToTensor())
            idx = 5
        if self.dataset == 'Curves':
            # n = 10,000 and p = 784
            data = loadmat(self.directory + "/curves.mat")
            X = torch.Tensor(data['X_test']).transpose(0,1)
            Y = torch.Tensor(data['y_test']).transpose(0,1).reshape(-1)
            idx = 0 # we only use Curves for Autoencoder
            return X, Y, idx
        if self.dataset == 'EMNIST_digits':
            # n = 10,000 and p = 784
            data = datasets.EMNIST(self.directory, 'digits', train=False, download=True, transform=transforms.ToTensor())
            idx = 5
        # the following converts it to a [n,d] tensor
        n = len(data)
        data = torch.utils.data.DataLoader(data, batch_size=n) # one batch contains all the data
        X, Y = iter(data).next() # extract the one batch
        # X is in the form [n, num_channels, ?, ?]
        X = X.reshape(n,-1)
        return X, Y, idx
