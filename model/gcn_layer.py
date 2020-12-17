import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_module import BaseModule

class GCNLayer(BaseModule):
    def __init__(self, config, laplacian):
        super().__init__(config)
        self.laplacian = laplacian
        self.Wv = nn.Linear(config.embedding_dim, config.encoding_dim)
    
    def forward(self, embeddings, indexes):
        x = torch.mm(self.laplacian, self.Wv(embeddings))
        outs = x.index_select(0, indexes)
        outs = torch.relu(outs)
        return outs