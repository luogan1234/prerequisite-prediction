import numpy as np
import os
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader, random_split
import json
import tqdm
import random
import torch
import pickle

class PreqDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.dataset_path = 'dataset/{}/'.format(config.dataset)
        self.config = config
        with open(os.path.join(self.dataset_path, 'concepts.txt'), 'r', encoding='utf-8') as f:
            self.concepts = [c for c in f.read().split('\n') if c]
        # convert concepts into token ids by BertTokenizer
        file = os.path.join(self.dataset_path, 'concept_tokens.pkl')
        if not os.path.exists(file):
            if config.language == 'en':
                tokenzier = BertTokenizer.from_pretrained('bert-base-uncased')
            if config.language == 'zh':
                tokenzier = BertTokenizer.from_pretrained('bert-base-chinese')
            self.tokens = []
            for concept in self.concepts:
                token = tokenzier.encode(concept, truncation=True, max_length=config.max_term_length)
                self.tokens.append(token)
            with open(file, 'wb') as f:
                pickle.dump(self.tokens, f)
        else:
            with open(file, 'rb') as f:
                self.tokens = pickle.load(f)
        # get concept embeddings by BertModel
        file = os.path.join(self.dataset_path, 'embeddings.pth')
        if not os.path.exists(file):
            if config.language == 'en':
                bert = BertModel.from_pretrained('bert-base-uncased')
            if config.language == 'zh':
                bert = BertModel.from_pretrained('bert-base-chinese')
            for p in bert.parameters():
                p.requires_grad = False
            bert.eval()
            bert.to(config.device)
            concept_embedding, token_embedding = [], []
            for token in tqdm.tqdm(self.tokens):
                token = torch.tensor(token, dtype=torch.long).to(config.device)
                with torch.no_grad():
                    h, _ = bert(token.unsqueeze(0))
                    h = h.squeeze(0)[1:-1]
                    ce = torch.mean(h, 0)
                te = torch.zeros((self.config.max_term_length, h.size(1)))
                te[:h.size(0), :] = h
                concept_embedding.append(ce.cpu())
                token_embedding.append(te)
            pca = PCA()
            X = torch.stack(concept_embedding).numpy()
            X = pca.fit_transform(X)
            self.concept_embedding = torch.from_numpy(X)
            X = torch.cat(token_embedding, 0).cpu().numpy()
            pca.fit(X)
            tmp = []
            for te in token_embedding:
                X = te.cpu().numpy()
                tmp.append(torch.from_numpy(pca.transform(X)))
            self.token_embedding = torch.stack(tmp)
            with open(file, 'wb') as f:
                torch.save([self.concept_embedding, self.token_embedding], f)
        else:
            with open(file, 'rb') as f:
                self.concept_embedding, self.token_embedding = torch.load(f)
        print('Get concept embeddings done.')
        # load prerequisite pairs
        self.data = []
        with open(os.path.join(self.dataset_path, 'pairs.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                s = line.split('\t')
                i1, i2, label = self.concepts.index(s[0]), self.concepts.index(s[1]), int(s[2])
                t1, t2 = self.tokens[i1], self.tokens[i2]
                self.data.append({'i1': i1, 'i2': i2, 't1': t1, 't2': t2, 'label': label})
        n = len(self.concepts)
        graph_path = os.path.join(self.dataset_path, 'graph.npy')
        if os.path.exists(graph_path):
            self.graphs = np.load(graph_path)
        else:
            self.graphs = [np.eye(n)]
        feature_path = os.path.join(self.dataset_path, 'feature.npy')
        if os.path.exists(feature_path):
            self.feature = np.load(feature_path)
        else:
            self.feature = np.zeros((0, n, n))
        print('data loader init finished.')
    
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
                    t1, t2 = self.tokens[i], self.tokens[j]
                    feature = self.feature[:, i, j]
                    data.append({'i1': i, 'i2': j, 't1': t1, 't2': t2, 'f': feature, 'label': match[i][j]})
        return match, data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx].copy()
        datum['f'] = self.feature[:, datum['i1'], datum['i2']]
        return datum

class MyBatch:
    def __init__(self, config):
        self.config = config
    
    def __call__(self, data):
        max_len_t1, max_len_t2 = 0, 0
        for datum in data:
            max_len_t1 = max(max_len_t1, len(datum['t1']))
            max_len_t2 = max(max_len_t2, len(datum['t2']))
        i1, i2, t1, t2, f, labels = [], [], [], [], [], []
        for datum in data:
            i1.append(datum['i1'])
            i2.append(datum['i2'])
            t1.append(datum['t1']+[0]*(max_len_t1-len(datum['t1'])))
            t2.append(datum['t2']+[0]*(max_len_t2-len(datum['t2'])))
            f.append(datum['f'])
            labels.append(datum['label'])
        i1 = torch.tensor(i1, dtype=torch.long).to(self.config.device)
        i2 = torch.tensor(i2, dtype=torch.long).to(self.config.device)
        t1 = torch.tensor(t1, dtype=torch.long).to(self.config.device)
        t2 = torch.tensor(t2, dtype=torch.long).to(self.config.device)
        f = torch.tensor(f, dtype=torch.float).to(self.config.device)
        obj = {'i1': i1, 'i2': i2, 't1': t1, 't2': t2, 'f': f, 'labels': labels}
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
        train = DataLoader(splits[0], self.config.batch_size('train'), shuffle=True, collate_fn=self.fn)
        eval = DataLoader(splits[1], self.config.batch_size('eval'), shuffle=False, collate_fn=self.fn)
        test = DataLoader(splits[2], self.config.batch_size('eval'), shuffle=False, collate_fn=self.fn)
        return train, eval, test
    
    def get_predict(self):
        match, data = self.dataset.all_pairs()
        data = DataLoader(data, self.config.batch_size('eval'), shuffle=False, collate_fn=self.fn)
        return match, data