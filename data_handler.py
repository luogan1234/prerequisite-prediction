import numpy as np
import os
import nltk
import json
import tqdm

class DataHandler:
    def __init__(self, dataset, model_name, lang, concat_feature, use_wiki, max_term_length, max_sentence_length):
        self.dataset = dataset
        self.max_term_length = max_term_length
        self.max_sentence_length = max_sentence_length
        self.model_name = model_name
        self.lang = lang
        self.concat_feature = concat_feature
        self.use_wiki = use_wiki
        dataset_path = 'dataset/{}/'.format(dataset)
        self.vocabs, self.concepts = [], []
        with open(os.path.join(dataset_path, 'vocabs.txt'), 'r', encoding='utf-8') as f:
            for line in f.read().split('\n'):
                if line:
                    self.vocabs.append(line)
        self.embeddings = np.load(os.path.join(dataset_path, 'embeddings.npy'))
        self.concept_embeddings = []
        with open(os.path.join(dataset_path, 'concepts.txt'), 'r', encoding='utf-8') as f:
            for line in f.read().split('\n'):
                if line:
                    self.concepts.append(line)
                    self.concept_embeddings.append(self.embeddings[self.vocabs.index(line)])
        self.concept_embeddings = np.array(self.concept_embeddings)
        if self.use_wiki:
            self.concept_text = {}
            with open(os.path.join(dataset_path, 'concept_abstract.json'), 'r', encoding='utf-8') as f:
                for line in f.read().split('\n'):
                    if line:
                        obj = json.loads(line)
                        text = obj['text']
                        if self.lang == 'en':
                            text = text.split(' ')
                        if self.lang == 'zh':
                            text = list(text)
                        text = text[:self.max_sentence_length]
                        self.concept_text[obj['concept']] = self.to_index(text)
        self.n = 0
        self.inputs, self.ids, self.texts, self.labels = [], [], [], []
        self.class_indexes = {}
        self.num_classes = 0
        with open(os.path.join(dataset_path, 'preq.txt'), 'r', encoding='utf-8') as f:
            for line in f.read().split('\n'):
                if line:
                    s = line.split('\t')
                    assert len(s) == 3
                    if self.use_wiki:
                        if s[0] not in self.concept_text or s[1] not in self.concept_text:
                            continue
                        text = [self.concept_text[s[0]], self.concept_text[s[1]]]
                        self.texts.append(text)
                    input = [self.to_index(s[0]), self.to_index(s[1])]
                    id = [self.concepts.index(s[0]), self.concepts.index(s[1])]
                    label = int(s[2])
                    self.num_classes = max(self.num_classes, label+1)
                    if label not in self.class_indexes:
                        self.class_indexes[label] = []
                    self.class_indexes[label].append(self.n)
                    self.inputs.append(input)
                    self.ids.append(id)
                    self.labels.append(label)
                    self.n += 1
        feature_path = os.path.join(dataset_path, 'features.npy')
        if os.path.exists(feature_path) and self.concat_feature:
            self.features = np.load(feature_path)
        else:
            self.features = np.zeros((self.n, 0), dtype=np.float32)
        graph_path = os.path.join(dataset_path, 'graph.npy')
        if os.path.exists(graph_path):
            self.graph = np.load(graph_path)
        else:
            self.graph = np.zeros((self.n, self.n), dtype=np.float32)
        print('data handler init finished.')
    
    def to_index(self, text):
        if isinstance(text, str):
            if self.lang == 'en':
                text = nltk.tokenize.word_tokenize(text.lower())
            if self.lang == 'zh':
                text = list(text)
        vec = [self.vocabs.index(w) for w in text]
        return vec
    
    def padding(self, vec, length):
        if len(vec) > length:
            vec = vec[:length]
        else:
            vec.extend([self.to_index(['PAD'])[0]] * (length - len(vec)))
        vec = np.array(vec, dtype=np.int64)
        return vec
    
    def prepare_data(self, split, total_splits):
        assert split in range(total_splits)
        train, eval = [], []
        for s in range(total_splits):
            group = []
            for i in self.class_indexes:
                n = len(self.class_indexes[i])
                split_size = n // total_splits
                for j in range(s*split_size, (s+1)*split_size):
                    index = self.class_indexes[i][j]
                    if not self.use_wiki:
                        i0 = self.padding(self.inputs[index][0], self.max_term_length)
                        i1 = self.padding(self.inputs[index][1], self.max_term_length)
                    else:
                        i0 = self.padding(self.texts[index][0], self.max_sentence_length)
                        i1 = self.padding(self.texts[index][1], self.max_sentence_length)
                    i2 = np.array(self.ids[index][0], dtype=np.int64)
                    i3 = np.array(self.ids[index][1], dtype=np.int64)
                    f = self.features[index]
                    if self.model_name == 'LSTM':
                        group.append([[i0, i1, f], i])
                    if self.model_name == 'TextCNN':
                        group.append([[i0+self.to_index(['SEP'])+i1, f], i])
                    if self.model_name == 'GCN':
                        group.append([[i2, i3, f], i])
                    if self.model_name == 'GCN_LSTM':
                        group.append([[i0, i1, i2, i3, f], i])
                    if self.model_name == 'MLP':
                        group.append([[i2, i3, f], i])
            if s != split:
                train.extend(group)
            else:
                eval = group
        return train, eval