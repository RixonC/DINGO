import torch
import torch.nn as nn


class softmaxModel(nn.Module):
    """The model used in the softmax cross-entropy minimization problem.
    
    Arguments:
        num_features (int): The number of features per sample.
        num_classes (int): The number of classes.
    """
    
    def __init__(self, num_features, num_classes):
        super(softmaxModel, self).__init__()
        self.num_features = num_features
        self.num_classes  = num_classes
        self.W = nn.Parameter(torch.zeros(num_features, num_classes, 
                                          requires_grad=True))


    def forward(self, x):
        """Forward pass.
        
        Arguments:
            x (torch.Tensor): Input tensor.
        """
        X = x.reshape(-1, self.num_features)
        XW = torch.mm(X, self.W)
        return XW # use criterion: nn.CrossEntropyLoss()