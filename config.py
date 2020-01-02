import numpy as np
import torch

class Config:
    def __init__(self, store):
        self.concat_feature = store.concat_feature
        self.use_wiki = store.use_wiki
        
        self.embeddings = store.embeddings
        self.gcn_embeddings = store.gcn_embeddings
        self.embedding_dim = store.embeddings.shape[1]
        self.feature_dim = store.features.shape[1]
        self.graph = store.graph
        self.epochs = 30
        self.batch_size = 64
        self.lr = 1e-3
        self.num_classes = store.num_classes
        self.use_gpu = True
        self.total_splits = 5
        self.save_dir = None
        
        self.dropout = 0.5
        self.filter_sizes = (2, 3, 4)  # CNN
        self.num_filters = 256  # CNN
        self.hidden_size = 128  # RNN
        self.num_layers = 2  # RNN
        self.gcn_hidden = 256  # GCN
    
    def to_torch(self, x):
        if self.use_gpu:
            return torch.from_numpy(x).cuda()
        else:
            return torch.from_numpy(x)