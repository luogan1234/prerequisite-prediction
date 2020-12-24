import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_module import BaseModule
from model.lstm_encoder import LSTMEncoder1, LSTMEncoder2

class TextModule(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        if config.model == 'lstm1':
            self.encoder1 = LSTMEncoder1(config)
            self.encoder2 = LSTMEncoder1(config)
        if config.model == 'lstm2':
            self.encoder1 = LSTMEncoder2(config)
            self.encoder2 = LSTMEncoder2(config)
        self.out_dim = config.encoding_dim*2
    
    def forward(self, batch):
        if self.config.model == 'lstm1':
            t1, t2 = batch['t1'], batch['t2']
            e1, e2 = self.encoder1(t1), self.encoder2(t2)
        if self.config.model == 'lstm2':
            i1, i2 = batch['i1'], batch['i2']
            e1, e2 = self.encoder1(i1), self.encoder2(i2)
        outs = torch.cat([e1, e2], 1)
        return outs