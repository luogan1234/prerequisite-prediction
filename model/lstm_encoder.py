import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_module import BaseModule

class LSTMEncoder1(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.token_embedding = nn.Embedding(self.config.vocab_num, self.config.embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(self.config.embedding_dim, self.config.encoding_dim, batch_first=True, bidirectional=False)
    
    def forward(self, t, r):
        h = self.token_embedding(t)
        o, _ = self.lstm(h)
        outs = torch.stack([o[i, r[i], :] for i in range(len(r))])
        return outs

class LSTMEncoder2(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.lstm = nn.LSTM(self.config.embedding_dim, self.config.encoding_dim, batch_first=True, bidirectional=False)
    
    def forward(self, i, r):
        h = self.config.token_embedding.index_select(0, i)
        o, _ = self.lstm(h)
        outs = torch.stack([o[i, r[i], :] for i in range(len(r))])
        return outs