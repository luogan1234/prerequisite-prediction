import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_module import BaseModule
from model.gcn_layer import GCNLayer
from model.gat_layer import GatLayer

class GraphModule(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.layers1, self.layers2 = [], []
        for laplacian in config.laplacians1:
            layer = self.new_layer(config, laplacian)
            self.layers1.append(layer)
        for laplacian in config.laplacians2:
            layer = self.new_layer(config, laplacian)
            self.layers2.append(layer)
        self.layers1, self.layers2 = nn.ModuleList(self.layers1), nn.ModuleList(self.layers2)
        self.gcn_number = len(config.laplacians1)+len(config.laplacians2)
        self.out_dim = config.encoding_dim*self.gcn_number
    
    def new_layer(self, config, laplacian):
        if self.config.model == 'gcn':
            layer = GCNLayer(config, laplacian)
        if self.config.model == 'gat':
            layer = GatLayer(config, laplacian)
        assert layer, 'Layer in graph model is not supported.'
        return layer

    def forward(self, inputs):
        i1, i2 = inputs['i1'], inputs['i2']
        o1 = [layer(self.config.concept_embedding, i1) for layer in self.layers1]
        o2 = [layer(self.config.concept_embedding, i2) for layer in self.layers2]
        outs = torch.cat(o1+o2, -1)
        return outs