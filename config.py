import numpy as np
import torch

class Config:
    def __init__(self, dataset, model, concat_feature, embedding_dim, encoding_dim, info, seed, cpu):
        self.dataset = dataset
        self.model = model
        self.concat_feature = concat_feature
        self.embedding_dim = embedding_dim
        self.encoding_dim = encoding_dim
        self.info = info
        self.seed = seed
        self.device = 'cpu' if cpu else 'cuda'
        if dataset in ['moocen']:
            self.language = 'en'
            self.vocab_num = 30522  # bert-base-uncased
        if dataset in ['mooczh']:
            self.language = 'zh'
            self.vocab_num = 21128  # bert-base-chinese
        assert self.language, 'Need to provide the language information for new datasets'
        
        self.max_term_length = 20
        self.max_epochs = 500
        self.attention_dim = 32
        self.early_stop_time = 20
        self.lr = 1e-5 if model == 'bert' else 1e-3
        self.num_classes = 2
    
    def batch_size(self, mode):
        if mode == 'train':
            return 16
        else:
            return 128
    
    def set_parameters(self, dataset):
        gcn_number = dataset.graphs.shape[0]
        self.laplacians1 = [self.to_laplacian_matrix(dataset.graphs[i]).T for i in range(gcn_number)]
        self.laplacians2 = [self.to_laplacian_matrix(dataset.graphs[i]) for i in range(gcn_number)]
        self.feature_dim = dataset.feature.shape[0]
        self.concept_embedding = dataset.concept_embedding[:, :self.embedding_dim].to(self.device)
        self.token_embedding = dataset.token_embedding[:, :, :self.embedding_dim].to(self.device)
    
    def to_laplacian_matrix(self, graph):
        a = np.eye(graph.shape[0]) + graph
        d = np.power(np.sum(np.abs(a), 1), -1)
        d[np.isinf(d)] = 0
        d = np.diag(d)
        laplacian = np.array(np.matmul(d, a), dtype=np.float32)
        laplacian = torch.from_numpy(laplacian).to(self.device)
        return laplacian
    
    def store_name(self):
        return '{}_{}_{}_{}_{}_{}'.format(self.dataset, self.model, self.concat_feature, self.embedding_dim, self.encoding_dim, self.seed)
    
    def parameter_info(self):
        obj = {'dataset': self.dataset, 'model': self.model, 'concat_feature': self.concat_feature, 'embedding_dim': self.embedding_dim, 'encoding_dim': self.encoding_dim, 'info': self.info, 'seed': self.seed}
        return obj