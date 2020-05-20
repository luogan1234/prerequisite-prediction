import numpy as np
import torch

class Config:
    def __init__(self, store, feature_dim):
        self.embedding_dim = 36
        self.concept_embedding = store.concept_embedding[:, :self.embedding_dim]
        self.token_embedding = store.token_embedding[:, :, :self.embedding_dim]
        
        self.max_epochs = 500
        self.early_stop_time = 10
        self.min_check_epoch = 5
        self.batch_size = 32
        self.lr = 1e-3
        self.num_classes = 2
        self.total_splits = 10
        
        self.feature_dim = feature_dim
        self.filter_sizes = (2, 4)  # TextCNN
        assert feature_dim % len(self.filter_sizes) == 0
        gcn_number = store.graph.shape[0]
        self.laplacians1 = [self.to_laplacian_matrix(store.graph[i, :, :]).T for i in range(gcn_number)]
        self.laplacians2 = [self.to_laplacian_matrix(store.graph[i, :, :]) for i in range(gcn_number)]
    
    def to_laplacian_matrix(self, graph):
        a = np.eye(graph.shape[0]) + graph
        d = np.power(np.sum(np.abs(a), 1), -1)
        d[np.isinf(d)] = 0
        d = np.diag(d)
        laplacian = np.array(np.matmul(d, a), dtype=np.float32)
        laplacian = torch.from_numpy(laplacian).cuda()
        return laplacian