# -*- coding: utf-8 -*-

import torch


def GMM_data(num_train_samples, num_train_features, num_test_samples, seed=0):
    '''Return the data and parameters used for the GMM experiment.

    Args:
        num_train_samples: Number of training samples.
        num_train_features: Number of training features.
        num_test_samples: Number of test samples.
        seed: set manual random seed.
    '''
    torch.manual_seed(seed)

    p = num_train_features

    t = torch.Tensor([1]).reshape(1,1)
    alpha = (torch.tanh(t)+1)/2
    mu1 = -torch.rand(p,1)
    mu2 = torch.rand(p,1)
    true_weights = torch.cat([t,mu1,mu2])

    A1 = torch.randn(p,p)
    A2 = torch.randn(p,p)

    Q1 = torch.qr(A1)[0]
    Q2 = torch.qr(A2)[0]

    D1 = torch.diag(torch.logspace(0,1,steps=p))
    D2 = torch.diag(torch.logspace(1,0,steps=p))

    C1 = 0.32*torch.mm(D1,Q1)
    C2 = 0.32*torch.mm(D2,Q2)
    c1 = torch.det(C1)
    c2 = torch.det(C2)

    Sigma1 = torch.inverse(torch.mm(C1.transpose(0,1),C1))
    Sigma2 = torch.inverse(torch.mm(C2.transpose(0,1),C2))

    ber = torch.distributions.Bernoulli(torch.tensor([1-alpha])) # (1-alpha) chance of 1; alpha chance of 0
    mn1 = torch.distributions.multivariate_normal.MultivariateNormal(mu1.reshape(-1),Sigma1)
    mn2 = torch.distributions.multivariate_normal.MultivariateNormal(mu2.reshape(-1),Sigma2)

    train_xs = []
    for k in range(num_train_samples):
        i = ber.sample()
        if i == 0:
            train_xs.append(mn1.sample().reshape(1,-1))
        else:
            train_xs.append(mn2.sample().reshape(1,-1))
    train_X = torch.cat(train_xs, dim=0)

    test_xs = []
    for k in range(num_test_samples):
        i = ber.sample()
        if i == 0:
            test_xs.append(mn1.sample().reshape(1,-1))
        else:
            test_xs.append(mn2.sample().reshape(1,-1))
    test_X = torch.cat(test_xs, dim=0)

    return train_X, test_X, C1, C2, c1, c2, true_weights


if __name__ == "__main__":
    train_X, test_X, C1, C2, c1, c2, true_weights = GMM_data(10000, 50, 10)
    print(c1,c2)
