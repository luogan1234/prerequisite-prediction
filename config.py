import numpy as np
import torch

class Config:
    def __init__(self, dataset, model, feature_dim, result_path, save_model, seed, cpu):
        self.dataset = dataset
        self.model = model
        self.feature_dim = feature_dim
        self.result_path = result_path
        self.save_model = save_model
        self.seed = seed
        self.device = 'cpu' if cpu else 'cuda'
        if dataset in ['moocen']:
            self.language = 'en'
        if dataset in ['mooczh']:
            self.language = 'zh'
        
        self.max_term_length = 7
        self.embedding_dim = 36
        self.max_epochs = 500
        self.early_stop_time = 10
        self.min_check_epoch = 5
        self.batch_size = 8
        self.lr = 1e-3
        self.num_classes = 2
        
        self.filter_sizes = (2, 4)  # TextCNN
        assert feature_dim % len(self.filter_sizes) == 0
    
    def set_parameters(self, dataset):
        gcn_number = dataset.graphs.shape[0]
        self.laplacians1 = [self.to_laplacian_matrix(dataset.graphs[i]).T for i in range(gcn_number)]
        self.laplacians2 = [self.to_laplacian_matrix(dataset.graphs[i]) for i in range(gcn_number)]
        self.user_feature_dim = dataset.user_feature.shape[0]
        self.concept_embedding = dataset.concept_embedding[:, :self.embedding_dim]
        self.token_embedding = dataset.token_embedding[:, :, :self.embedding_dim]
    
    def to_laplacian_matrix(self, graph):
        a = np.eye(graph.shape[0]) + graph
        d = np.power(np.sum(np.abs(a), 1), -1)
        d[np.isinf(d)] = 0
        d = np.diag(d)
        laplacian = np.array(np.matmul(d, a), dtype=np.float32)
        laplacian = torch.from_numpy(laplacian).to(self.device)
        return laplacian
    
    def model_name(self):
        return '{}_{}_{}_{}'.format(self.dataset, self.model, self.feature_dim, self.seed)
    
    def parameter_info(self):
        obj = {'dataset': self.dataset, 'model': self.model, 'feature_dim': self.feature_dim, 'seed': self.seed}
        return obj