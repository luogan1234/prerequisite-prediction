import numpy as np
import os
from pytorch_pretrained_bert import BertTokenizer, BertModel
from sklearn.decomposition import PCA
import json
import tqdm
import random
import torch

class DataLoader:
    def __init__(self, dataset, model_name, lang):
        self.dataset = dataset
        self.max_term_length = 7
        self.model_name = model_name
        self.lang = lang
        dataset_path = 'dataset/{}/'.format(dataset)
        with open(os.path.join(dataset_path, 'concepts.txt'), 'r', encoding='utf-8') as f:
            self.concepts = [c for c in f.read().split('\n') if c]
        self.load_embedding()
        self.inputs = [[], []]
        with open(os.path.join(dataset_path, 'preq.txt'), 'r', encoding='utf-8') as f:
            data = [line for line in f.read().split('\n') if line]
            for line in data:
                if line:
                    s = line.split('\t')
                    self.inputs[int(s[2])].append({'i1': self.concepts.index(s[0]), 'i2': self.concepts.index(s[1])})
        n = len(self.concepts)
        graph_path = os.path.join(dataset_path, 'graph.npy')
        if os.path.exists(graph_path):
            self.graph = np.load(graph_path)
        else:
            self.graph = np.eye(n)
        feature_path = os.path.join(dataset_path, 'user_feature.npy')
        if os.path.exists(feature_path):
            self.user_feature = np.load(feature_path)
        else:
            self.user_feature = np.zeros((0, n, n))
        print('data loader init finished.')
    
    def load_embedding(self):
        file = 'dataset/{}/embedding.pth'.format(self.dataset)
        if os.path.exists(file):
            with open(file, 'rb') as f:
                self.concept_embedding, self.token_embedding = torch.load(f)
        else:
            print('load embedding:')
            self.concept_embedding, self.token_embedding = [], []
            if self.lang == 'en':
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                bert = BertModel.from_pretrained('bert-base-uncased')
            if self.lang == 'zh':
                tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
                bert = BertModel.from_pretrained('bert-base-chinese')
            for p in bert.parameters():
                p.requires_grad = False
            bert.cuda()
            for concept in tqdm.tqdm(self.concepts):
                tokens = ['[CLS]']+tokenizer.tokenize(concept)+['[SEP]']
                ids = tokenizer.convert_tokens_to_ids(tokens)
                ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).cuda()
                with torch.no_grad():
                    h, _ = bert(ids, output_all_encoded_layers=False)
                    h = h.squeeze(0)[1:-1, :]
                    ce = torch.mean(h, 0)
                    h = h[:self.max_term_length, :]
                te = torch.zeros((self.max_term_length, h.size(1)))
                te[:h.size(0), :] = h
                self.concept_embedding.append(ce)
                self.token_embedding.append(te)
            pca = PCA()
            X = torch.stack(self.concept_embedding).cpu().numpy()
            X = pca.fit_transform(X)
            self.concept_embedding = torch.from_numpy(X)
            pca = PCA()
            X = torch.cat(self.token_embedding, 0).cpu().numpy()
            pca.fit(X)
            tmp = []
            for token_embedding in self.token_embedding:
                X = token_embedding.cpu().numpy()
                tmp.append(torch.from_numpy(pca.transform(X)))
            self.token_embedding = torch.stack(tmp)
            with open(file, 'wb') as f:
                torch.save([self.concept_embedding, self.token_embedding], f)
    
    def prepare_data(self, eval_split, total_splits):
        assert eval_split in range(total_splits)
        test_split = (eval_split + 1) % total_splits
        train, eval, test = [], [], []
        for i in range(total_splits):
            group = []
            for label in range(2):
                n = len(self.inputs[label])
                split_size = (n-1) // total_splits + 1
                for input in self.inputs[label][i*split_size: (i+1)*split_size]:
                    feature = self.user_feature[:, input['i1'], input['i2']]
                    group.append({'i1': input['i1'], 'i2': input['i2'], 'f': feature, 'label': label})
            if i not in [eval_split, test_split]:
                train.extend(group)
            if i == eval_split:
                eval = group
            if i == test_split:
                test = group
        return train, eval, test
    
    def prepare_all_pair(self):
        data = []
        n = len(self.concepts)
        match = [[-1]*n for i in range(n)]
        for label in range(len(self.inputs)):
            for input in self.inputs[label]:
                match[input['i1']][input['i2']] = label
        for i in range(n):
            for j in range(n):
                if i != j:
                    feature = self.user_feature[:, i, j]
                    data.append({'i1': i, 'i2': j, 'f': feature, 'label': match[i][j]})
        return data