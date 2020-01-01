import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(config.embeddings), freeze=True)
    
    def forward(self, data):
        raise NotImplementedError