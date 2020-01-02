import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base_model import BaseModel
from models.gcn_layer import GraphConvolution

class GCN(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.graph = config.to_torch(config.graph)
        self.h0 = config.to_torch(config.gcn_embeddings)
        self.gc1 = GraphConvolution(config.embedding_dim, config.gcn_hidden, self.graph)
        self.gc2 = GraphConvolution(config.gcn_hidden, config.gcn_hidden, self.graph)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.gcn_hidden*2+config.feature_dim, config.num_classes)

    def forward(self, inputs):
        x1, x2, features = inputs
        x = F.relu(self.gc1(self.h0))
        x = self.gc2(self.dropout(x))
        o1 = torch.index_select(x, 0, x1)
        o2 = torch.index_select(x, 0, x2)
        o = torch.cat([o1, o2, features], -1)
        out = self.fc(o)
        print(out)
        return out