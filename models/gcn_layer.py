import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, laplacians):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gcn_number = len(laplacians)
        self.laplacians = laplacians
        w = torch.empty(self.gcn_number, in_features, out_features)
        std = 1. / math.sqrt(out_features)
        nn.init.uniform_(w, -std, std)
        self.weights = nn.Parameter(w)
    
    def forward(self, input, index):
        out = []
        for i in range(self.gcn_number):
            x = torch.mm(self.laplacians[i], torch.mm(input, self.weights[i]))
            x = x.index_select(0, index)
            out.append(x)
        out = torch.cat(out, -1)
        return out
    
    def __repr__(self):
        return '{} ({} -> {})'.format(self.__class__.__name__, self.in_features, self.out_features)