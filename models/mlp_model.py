import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base_model import BaseModel
from models.mlp_classification_layer import MLPClassification

class MLP(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.concept_embedding = nn.Embedding.from_pretrained(torch.tensor(config.concept_embeddings), freeze=True)
        self.fc1 = nn.Linear(config.embedding_dim, config.mlp_dim)
        self.fc2 = nn.Linear(config.embedding_dim, config.mlp_dim)
        self.fc = MLPClassification(config.mlp_dim*2, config.num_classes)

    def forward(self, inputs):
        x1, x2 = inputs
        o1 = self.concept_embedding(x1)
        o1 = F.relu(self.fc1(o1))
        o2 = self.concept_embedding(x2)
        o2 = F.relu(self.fc2(o2))
        o = torch.cat([o1, o2], -1)
        out = self.fc(o)
        return out