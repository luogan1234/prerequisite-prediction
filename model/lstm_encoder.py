import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_module import BaseModule

class LSTMEncoder(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.word_embedding = nn.Embedding(self.config.vocab_num, self.config.embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(self.config.embedding_dim, self.config.encoding_dim, batch_first=True, bidirectional=False)
    
    def forward(self, t):
        h = self.word_embedding(t)
        o, _ = self.lstm(h)
        outs = o[:, -1, :]
        return outs