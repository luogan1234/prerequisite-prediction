import numpy as np
import os
from pytorch_pretrained_bert import BertTokenizer, BertModel
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader, random_split
import json
import tqdm
import random
import torch

class PreqDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.dataset_path = 'dataset/{}/'.format(config.dataset)
        self.config = config
        with open(os.path.join(self.dataset_path, 'concepts.txt'), 'r', encoding='utf-8') as f:
            self.concepts = [c for c in f.read().split('\n') if c]
        self.load_embedding()
        self.data = []
        with open(os.path.join(self.dataset_path, 'preq.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                s = line.split('\t')
                self.data.append({'i1': self.concepts.index(s[0]), 'i2': self.concepts.index(s[1]), 'label': int(s[2])})
        n = len(self.concepts)
        graph_path = os.path.join(self.dataset_path, 'graph.npy')
        if os.path.exists(graph_path):
            self.graphs = np.load(graph_path)
        else:
            self.graphs = [np.eye(n)]
        feature_path = os.path.join(self.dataset_path, 'user_feature.npy')
        if os.path.exists(feature_path):
            self.user_feature = np.load(feature_path)
        else:
            self.user_feature = np.zeros((0, n, n))
        print('data loader init finished.')
    
    def load_embedding(self):
        file = os.path.join(self.dataset_path, 'embedding.pth')
        if os.path.exists(file):
            with open(file, 'rb') as f:
                self.concept_embedding, self.token_embedding = torch.load(f)
        else:
            print('load embedding:')
            if self.config.language == 'en':
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                bert = BertModel.from_pretrained('bert-base-uncased')
            if self.config.language == 'zh':
                tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
                bert = BertModel.from_pretrained('bert-base-chinese')
            for p in bert.parameters():
                p.requires_grad = False
            cls, sep = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])  # [CLS]: 101, [SEP]: 102
            concept_embedding, token_embedding = [], []
            bert.eval()
            bert.to(self.config.device)
            for concept in tqdm.tqdm(self.concepts):
                ids = tokenizer.tokenize(concept)
                ids = tokenizer.convert_tokens_to_ids(ids)[:self.config.max_term_length]
                ids = torch.tensor([cls]+ids+[sep], dtype=torch.long).unsqueeze(0).to(self.config.device())
                with torch.no_grad():
                    h, _ = bert(ids, output_all_encoded_layers=False)
                    h = h.squeeze(0)[1:-1]
                    ce = torch.mean(h, 0)
                te = torch.zeros((self.config.max_term_length, h.size(1)))
                te[:h.size(0), :] = h
                concept_embedding.append(ce)
                token_embedding.append(te)
            pca = PCA()
            X = torch.stack(concept_embedding).cpu().numpy()
            X = pca.fit_transform(X)
            self.concept_embedding = torch.from_numpy(X)
            pca = PCA()
            X = torch.cat(token_embedding, 0).cpu().numpy()
            pca.fit(X)
            tmp = []
            for te in token_embedding:
                X = te.cpu().numpy()
                tmp.append(torch.from_numpy(pca.transform(X)))
            self.token_embedding = torch.stack(tmp)
            with open(file, 'wb') as f:
                torch.save([self.concept_embedding, self.token_embedding], f)
    
    def all_pairs(self):
        n = len(self.concepts)
        match = np.empty((n, n))
        match.fill(-1)
        for datum in self.data:
            match[datum['i1']][datum['i2']] = datum['label']
        data = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    feature = self.user_feature[:, i, j]
                    data.append({'i1': i, 'i2': j, 'f': feature, 'label': match[i][j]})
        return match, data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        datum['f'] = self.user_feature[:, datum['i1'], datum['i2']]
        return datum

class MyBatch:
    def __init__(self, config):
        self.config = config
    
    def __call__(self, data):
        i1 = [datum['i1'] for datum in data]
        i1 = torch.tensor(i1, dtype=torch.long).to(self.config.device)
        i2 = [datum['i2'] for datum in data]
        i2 = torch.tensor(i2, dtype=torch.long).to(self.config.device)
        f = [datum['f'] for datum in data]
        f = torch.tensor(f, dtype=torch.float).to(self.config.device)
        labels = [datum['label'] for datum in data]
        obj = {'i1': i1, 'i2': i2, 'f': f, 'labels': labels}
        return obj

class PreqDataLoader:
    def __init__(self, config):
        self.dataset = PreqDataset(config)
        config.set_parameters(self.dataset)
        self.config = config
        self.fn = MyBatch(config)
    
    def get_train(self):
        n = len(self.dataset)
        d = n // 10
        splits = random_split(self.dataset, [d*8, d, n-d*9])
        train = DataLoader(splits[0], self.config.batch_size, shuffle=True, collate_fn=self.fn)
        eval = DataLoader(splits[1], self.config.batch_size, shuffle=False, collate_fn=self.fn)
        test = DataLoader(splits[2], self.config.batch_size, shuffle=False, collate_fn=self.fn)
        return train, eval, test
    
    def get_predict(self):
        match, data = self.dataset.all_pairs()
        data = DataLoader(data, self.config.batch_size, shuffle=False, collate_fn=self.fn)
        return match, data