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
        self.gc1 = GraphConvolution(config.embedding_dim, config.gcn_hidden, config.laplacians1)
        self.gc2 = GraphConvolution(config.embedding_dim, config.gcn_hidden, config.laplacians2)
        self.gcn_number = len(config.laplacians1)+len(config.laplacians2)
        self.fc = MLPClassification(config.gcn_hidden*self.gcn_number, config.num_classes)

    def forward(self, inputs):
        x1, x2 = inputs
        o1 = F.relu(self.gc1(self.h0, x1))
        o2 = F.relu(self.gc2(self.h0, x2))
        o = torch.cat([o1, o2], -1)
        out = self.fc(o)
        return out