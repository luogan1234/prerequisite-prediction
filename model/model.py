import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_module import BaseModule
from model.text_module import TextModule
from model.graph_module import GraphModule
from model.mlp_layer import MLPLayer

class Model(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        if config.model in ['lstm', 'cnn', 'bert']:
            self.module = TextModule(config)
        if config.model in ['gcn', 'gat', 'gat2']:
            self.module = GraphModule(config)
        assert self.module, 'Module is not supported.'
        in_dim = self.module.out_dim+config.concat_user_feature*config.user_feature_dim
        self.fc = MLPLayer(in_dim, self.config.num_classes)
    
    def forward(self, batch):
        outs = self.module(batch)
        if self.config.concat_user_feature:
            outs = torch.cat([outs, batch['f']], -1)
        outs = self.fc(outs)
        return outs