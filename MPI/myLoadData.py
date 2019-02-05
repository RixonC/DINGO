import numpy as np
from scipy import sparse
from scipy.io import loadmat
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class myDataLoader(object):
    '''This class loads and processes data.

    Args:
        config: A dictionary of all necessary parameters.
    '''
    def __init__(self, config):
        self.dataset = config['dataset']
        self.directory = '../Data/' + config['dataset']
        self.objective_function = config['obj_fun']
        self.classes = config['classes']


    def process_data(self, X, Y, is_test_data=False):
        '''Process data for the given objective function.
        
        Args:
            X: Feature data.
            Y: Label data.
            is_test_data: A boolean indicating if this data will be used to compute test accuracy or error.
        '''
        n, d= X.shape
        if self.objective_function == 'softmax':
            I = np.ones(n)
            X_label = np.array([i for i in range(n)])
            Y = sparse.coo_matrix((I,(X_label, Y.numpy())), shape=(n, self.classes))
            Y = Y.tocsr().toarray()
            if is_test_data:
                Y = torch.Tensor(Y).type(torch.Tensor) # [n,C]
            else:
                Y = Y[:,:-1]
                Y = torch.Tensor(Y).type(torch.Tensor) # [n,C-1]
        return X.type(torch.Tensor), Y


    def load_train_data(self):
        '''Return the training data.'''
        if self.dataset == 'Curves':
            # n = 20,000 and p = 784
            data = loadmat(self.directory + "/curves.mat")
            X = torch.Tensor(data['X']).transpose(0,1)
            Y = torch.Tensor(data['y']).transpose(0,1).reshape(-1)
            X, Y = self.process_data(X, Y)
            return X, Y
        elif self.dataset == 'CIFAR10':
            # n = 50,000 and p = 3,072
            data = datasets.CIFAR10(self.directory, train=True, download=True, transform=transforms.ToTensor())
        else: # EMNIST_digits
            # n = 240,000 and p = 784
            data = datasets.EMNIST(self.directory, 'digits', train=True, download=True, transform=transforms.ToTensor())
        # the following converts it to a [n,-1] tensor
        n = len(data)
        data = torch.utils.data.DataLoader(data, batch_size=n) # one batch contains all the data
        X, Y = iter(data).next() # extract the one batch
        # X is in the form [n, num_channels, ?, ?]
        X = X.reshape(n,-1) # we leave Y as an array
        X, Y = self.process_data(X, Y)
        return X, Y


    def load_test_data(self):
        '''Return the test data.'''
        if self.dataset == 'Curves':
            # n = 10,000 and p = 784
            data = loadmat(self.directory + "/curves.mat")
            X = torch.Tensor(data['X_test']).transpose(0,1)
            Y = torch.Tensor(data['y_test']).transpose(0,1).reshape(-1)
            X, Y = self.process_data(X, Y, is_test_data=True)
            return X, Y
        if self.dataset == 'CIFAR10':
            # n = 10,000 and p = 3,072
            data = datasets.CIFAR10(self.directory, train=False, download=True, transform=transforms.ToTensor())
        if self.dataset == 'EMNIST_digits':
            # n = 40,000 and p = 784
            data = datasets.EMNIST(self.directory, 'digits', train=False, download=True, transform=transforms.ToTensor())
        # the following converts it to a [n,-1] tensor
        n = len(data)
        data = torch.utils.data.DataLoader(data, batch_size=n) # one batch contains all the data
        X, Y = iter(data).next() # extract the one batch
        # X is in the form [n, num_channels, ?, ?]
        X = X.reshape(n,-1)
        X, Y = self.process_data(X, Y, is_test_data=True)
        return X, Y
