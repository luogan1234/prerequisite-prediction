import numpy as np
import torch

class Config:
    def __init__(self, store, embedding_dim, feature_dim, max_epochs, use_cpu):
        self.use_gpu = not use_cpu
        
        self.embedding_dim = embedding_dim
        self.embeddings = store.embeddings[:, :embedding_dim]
        self.concept_embeddings = store.concept_embeddings[:, :embedding_dim]
        self.max_epochs = max_epochs
        self.early_stop_time = 50
        self.min_check_epoch = 100
        self.batch_size = 64
        self.lr = 1e-3
        self.num_classes = store.num_classes
        self.total_splits = 10
        
        self.feature_dim = feature_dim
        self.filter_sizes = (2, 4)  # CNN
        self.num_filters = self.feature_dim // len(self.filter_sizes) # CNN
        self.lstm_hidden = self.feature_dim // 2  # LSTM
        self.num_layers = 1  # LSTM
        self.gcn_hidden = self.feature_dim // 2  # GCN
        gcn_number = store.graph.shape[0]
        self.laplacians1 = [self.to_laplacian_matrix(store.graph[i, :, :]).T for i in range(gcn_number)]
        self.laplacians2 = [self.to_laplacian_matrix(store.graph[i, :, :]) for i in range(gcn_number)]
    
    def to_laplacian_matrix(self, graph):
        a = np.eye(graph.shape[0]) + graph
        d = np.power(np.sum(np.abs(a), 0), -1)
        d[np.isinf(d)] = 0
        d = np.diag(d)
        laplacian = np.matmul(d, a)
        laplacian = self.to_torch(np.array(laplacian, dtype=np.float32))
        return laplacian
    
    def to_torch(self, x):
        if self.use_gpu:
            return torch.from_numpy(x).cuda()
        else:
            return torch.from_numpy(x)