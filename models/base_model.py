import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.concept_embedding = config.concept_embedding.cuda()
        self.token_embedding = config.token_embedding.cuda()
    
    def forward(self, data):
        raise NotImplementedError