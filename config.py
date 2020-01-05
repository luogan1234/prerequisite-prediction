import numpy as np
import torch

class Config:
    def __init__(self, store):
        self.concat_feature = store.concat_feature
        self.use_wiki = store.use_wiki
        
        self.embeddings = store.embeddings
        self.concept_embeddings = store.concept_embeddings
        self.embedding_dim = store.embeddings.shape[1]
        self.feature_dim = store.features.shape[1]
        self.epochs = None
        self.batch_size = 64
        self.lr = 1e-3
        self.num_classes = store.num_classes
        self.use_gpu = True
        self.total_splits = 10
        self.save_dir = None
        
        self.dropout = 0.5
        self.filter_sizes = (3, 5)  # CNN
        self.num_filters = 128  # CNN
        self.lstm_hidden = 64  # RNN
        self.num_layers = 1  # RNN
        self.laplacian1 = self.to_laplacian_matrix(store.graph.T)
        self.laplacian2 = self.to_laplacian_matrix(store.graph)
        self.gcn_hidden = 128  # GCN
        self.mlp_dim = 128  # MLP
    
    def to_laplacian_matrix(self, graph):
        d = np.power(np.sum(graph, 0), -0.5)
        d[np.isinf(d)] = 0
        d = np.diag(d)
        a = graph + np.eye(graph.shape[0])
        laplacian = np.matmul(np.matmul(d, a), d)
        laplacian = self.to_torch(np.array(laplacian, dtype=np.float32))
        return laplacian
    
    def to_torch(self, x):
        if self.use_gpu:
            return torch.from_numpy(x).cuda()
        else:
            return torch.from_numpy(x)