import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_module import BaseModule
from model.lstm_encoder import LSTMEncoder

class TextModule(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        if config.model == 'lstm':
            self.encoder = LSTMEncoder(config)
        self.out_dim = config.encoding_dim*2
    
    def forward(self, batch):
        t1, t2 = batch['t1'], batch['t2']
        e1, e2 = self.encoder(t1), self.encoder(t2)
        outs = torch.cat([e1, e2], 1)
        return outs