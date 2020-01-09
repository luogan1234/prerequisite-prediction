import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base_model import BaseModel
from models.gcn_layer import GraphConvolution
from models.mlp_classification_layer import MLPClassification

class GCN(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.h0 = config.to_torch(config.concept_embeddings)
        self.gc1 = GraphConvolution(config.embedding_dim, config.gcn_hidden, config.laplacian1)
        self.gc2 = GraphConvolution(config.embedding_dim, config.gcn_hidden, config.laplacian2)
        self.fc = MLPClassification(config.gcn_hidden*2, config.num_classes)

    def forward(self, inputs):
        x1, x2 = inputs
        x = F.relu(self.gc1(self.h0))
        o1 = torch.index_select(x, 0, x1)
        x = F.relu(self.gc2(self.h0))
        o2 = torch.index_select(x, 0, x2)
        o = torch.cat([o1, o2], -1)
        out = self.fc(o)
        return out