import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base_model import BaseModel
from models.mlp_classification_layer import MLPClassification

class LSTM(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.lstm1 = nn.LSTM(config.embedding_dim, config.feature_dim, 1, bidirectional=False, batch_first=True)
        self.lstm2 = nn.LSTM(config.embedding_dim, config.feature_dim, 1, bidirectional=False, batch_first=True)
        self.fc = MLPClassification(config.feature_dim*2, config.num_classes)

    def forward(self, inputs):
        i1, i2 = inputs['i1'], inputs['i2']
        x1 = self.token_embedding.index_select(0, i1)
        x2 = self.token_embedding.index_select(0, i2)
        o1, _ = self.lstm1(x1)
        o1 = o1[:, -1, :]
        o2, _ = self.lstm2(x2)
        o2 = o2[:, -1, :]
        o = torch.cat([o1, o2], -1)
        out = self.fc(o)
        return out
