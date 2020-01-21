import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base_model import BaseModel
from models.mlp_classification_layer import MLPClassification

class LSTM(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.lstm1 = nn.LSTM(config.embedding_dim, config.lstm_hidden, config.num_layers, bidirectional=False, batch_first=True)
        self.lstm2 = nn.LSTM(config.embedding_dim, config.lstm_hidden, config.num_layers, bidirectional=False, batch_first=True)
        self.fc = MLPClassification(config.lstm_hidden*2, config.num_classes)

    def forward(self, inputs):
        x1, x2 = inputs
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        o1, _ = self.lstm1(x1)
        o1 = o1[:, -1, :]
        o2, _ = self.lstm2(x2)
        o2 = o2[:, -1, :]
        o = torch.cat([o1, o2], -1)
        out = self.fc(o)
        return out
