import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, laplacian):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.laplacian = laplacian

    def forward(self, input):
        x = torch.mm(self.laplacian, torch.mm(input, self.weight))
        return x
    
    def __repr__(self):
        return '{} ({} -> {})'.format(self.__class__.__name__, self.in_features, self.out_features)