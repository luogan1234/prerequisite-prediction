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
        self.gc1 = GraphConvolution(config.embedding_dim, config.feature_dim, config.laplacians1)
        self.gc2 = GraphConvolution(config.embedding_dim, config.feature_dim, config.laplacians2)
        self.gcn_number = len(config.laplacians1)+len(config.laplacians2)
        self.fc = MLPClassification(config.feature_dim*self.gcn_number, config.num_classes)

    def forward(self, inputs):
        i1, i2 = inputs['i1'], inputs['i2']
        o1 = F.relu(self.gc1(self.concept_embedding, i1))
        o2 = F.relu(self.gc2(self.concept_embedding, i2))
        o = torch.cat([o1, o2], -1)
        out = self.fc(o)
        return out