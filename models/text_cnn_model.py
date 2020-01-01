import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base_model import BaseModel

class TextCNN(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.convs = nn.ModuleList([nn.Conv2d(1, config.num_filters, (k, config.embedding_dim)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters*len(config.filter_sizes)+config.feature_dim, config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, inputs):
        x0, features = inputs
        x0 = self.embedding(x0)
        x0 = x0.unsqueeze(1)
        conv_out = torch.cat([self.conv_and_pool(x0, conv) for conv in self.convs], 1)
        conv_out = self.dropout(conv_out)
        out = torch.cat([conv_out, features], -1)
        out = self.fc(out)
        return out
