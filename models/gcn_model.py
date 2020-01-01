import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base_model import BaseModel

class GCN(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, x):
        return x