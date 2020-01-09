import numpy as np
import torch

class Config:
    def __init__(self, store):
        self.use_wiki = store.use_wiki
        
        self.embeddings = store.embeddings
        self.concept_embeddings = store.concept_embeddings
        self.embedding_dim = store.embeddings.shape[1]
        self.epochs = None
        self.batch_size = 64
        self.lr = 1e-3
        self.num_classes = store.num_classes
        self.use_gpu = True
        self.total_splits = 10
        self.save_dir = None
        
        self.dropout = 0.5
        self.filter_sizes = (3, 5)  # CNN
        self.num_filters = 64  # CNN
        self.lstm_hidden = 32  # RNN
        self.num_layers = 1  # RNN
        #self.graph = np.array(store.graph, dtype=np.float32)
        self.laplacian1 = self.to_laplacian_matrix(store.graph.T)
        self.laplacian2 = self.to_laplacian_matrix(store.graph)
        self.gcn_hidden = 64  # GCN
        self.mlp_dim = 64  # MLP
    
    def to_laplacian_matrix(self, graph):
        a = np.eye(graph.shape[0]) - graph
        d1 = np.power(np.sum(np.abs(a), 1), -0.5)
        d1[np.isinf(d1)] = 0
        d1 = np.diag(d1)
        d2 = np.power(np.sum(np.abs(a), 0), -0.5)
        d2[np.isinf(d2)] = 0
        d2 = np.diag(d2)
        laplacian = np.matmul(np.matmul(d1, a), d2)
        laplacian = self.to_torch(np.array(laplacian, dtype=np.float32))
        return laplacian
    
    def to_torch(self, x):
        if self.use_gpu:
            return torch.from_numpy(x).cuda()
        else:
            return torch.from_numpy(x)