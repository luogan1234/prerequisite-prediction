import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLPClassification(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        mid_features = in_features // 2
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, mid_features)
        self.fc2 = nn.Linear(mid_features, out_features)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input):
        input = self.dropout(input)
        x = F.relu(self.fc1(input))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def __repr__(self):
        return '{} ({} -> {})'.format(self.__class__.__name__, self.in_features, self.out_features)