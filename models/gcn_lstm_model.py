import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base_model import BaseModel
from models.gcn_layer import GraphConvolution
from models.mlp_classification_layer import MLPClassification

class GCN_LSTM(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.h0 = config.to_torch(config.concept_embeddings)
        self.gc1 = GraphConvolution(config.embedding_dim, config.gcn_hidden, config.laplacian1)
        self.gc2 = GraphConvolution(config.embedding_dim, config.gcn_hidden, config.laplacian2)
        self.lstm1 = nn.LSTM(config.embedding_dim, config.lstm_hidden, config.num_layers, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(config.embedding_dim, config.lstm_hidden, config.num_layers, bidirectional=True, batch_first=True)
        self.fc = MLPClassification(config.gcn_hidden*2+config.lstm_hidden*4, config.num_classes)

    def forward(self, inputs):
        x1, x2, x3, x4 = inputs
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        o1, _ = self.lstm1(x1)
        o1 = o1[:, -1, :]
        o2, _ = self.lstm2(x2)
        o2 = o2[:, -1, :]
        x = F.relu(self.gc1(self.h0))
        o3 = torch.index_select(x, 0, x3)
        x = F.relu(self.gc2(self.h0))
        o4 = torch.index_select(x, 0, x4)
        o = torch.cat([o1, o2, o3, o4], -1)
        out = self.fc(o)
        return out