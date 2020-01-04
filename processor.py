import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from sklearn import metrics
import os
import json
import numpy as np
import tqdm
import random
from models.lstm_model import LSTM
from models.text_cnn_model import TextCNN
from models.gcn_model import GCN
from models.gcn_lstm_model import GCN_LSTM
from models.mlp_model import MLP

def name_to_model(name, config):
    if name == 'LSTM':
        return LSTM(config)
    if name == 'TextCNN':
        return TextCNN(config)
    if name == 'GCN':
        return GCN(config)
    if name == 'GCN+LSTM':
        return GCN_LSTM(config)
    if name == 'MLP':
        return MLP(config)
    raise NotImplementedError

class Processor(object):
    def __init__(self, model_name, store, config):
        self.model_name = model_name
        self.store = store
        self.config = config
    
    def train_one_step(self, inputs, labels):
        labels = self.config.to_torch(labels)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def eval_one_step(self, inputs, labels):
        labels = self.config.to_torch(labels)
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, labels)
            predicts = torch.max(outputs.data, 1)[1].cpu().numpy()
        return predicts, loss
    
    def get_batch_data(self, data):
        inputs, labels = [], []
        for i in range(len(data[0][0])):
            inputs.append([])
        for sample in data:
            input, label = sample
            for i in range(len(input)):
                inputs[i].append(input[i])
            labels.append(label)
        inputs = [self.config.to_torch(np.array(input)) for input in inputs]
        labels = np.array(labels, dtype=np.int64)
        return inputs, labels
    
    def run_split(self, split):
        print("Split {} starts, use model {}".format(split, self.model_name))
        self.model = name_to_model(self.model_name, self.config)
        if self.config.use_gpu:
            self.model.cuda()
        best_para = self.model.state_dict()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        train, eval = self.store.prepare_data(split, self.config.total_splits)
        train_steps = (len(train) - 1) // self.config.batch_size + 1
        eval_steps = (len(eval) - 1) // self.config.batch_size + 1
        training_range = tqdm.tqdm(range(self.config.epochs))
        training_range.set_description("Epoch %d | loss: %.3f" % (0, 0))
        min_loss = 1e16
        for epoch in training_range:
            res = 0.0
            random.shuffle(train)
            for i in range(train_steps):
                inputs, labels = self.get_batch_data(train[i*self.config.batch_size: (i+1)*self.config.batch_size])
                loss = self.train_one_step(inputs, labels)
                res += loss
            training_range.set_description("Epoch %d | loss: %.3f" % (epoch, res))
            if res < min_loss:
                min_loss = res
                best_para = self.model.state_dict()
        self.model.load_state_dict(best_para)
        if self.config.save_dir:
            save_path = os.path.join(self.config.save_dir, '{}_{}.ckpt'.format(self.model_name, split))
            print("Split {} train finished, save at {}".format(split, self.config.save_dir))
            torch.save(best_para, save_path)
        print('Split {} train finished, min_loss {:.3f}'.format(split, min_loss))
        self.model.eval()
        labels_all = np.array([], dtype=int)
        predicts_all = np.array([], dtype=int)
        for i in range(eval_steps):
            inputs, labels = self.get_batch_data(eval[i*self.config.batch_size: (i+1)*self.config.batch_size])
            predicts, loss = self.eval_one_step(inputs, labels)
            labels_all = np.append(labels_all, labels)
            predicts_all = np.append(predicts_all, predicts)
        acc = metrics.accuracy_score(labels_all, predicts_all)
        p = metrics.precision_score(labels_all, predicts_all, average='weighted')
        r = metrics.recall_score(labels_all, predicts_all, average='weighted')
        f1 = metrics.f1_score(labels_all, predicts_all, average='weighted')
        report = metrics.classification_report(labels_all, predicts_all)
        confusion = metrics.confusion_matrix(labels_all, predicts_all)
        print('Split {} eval finished, acc {:.3f}, p {:.3f}, r {:.3f}, f1 {:.3f}'.format(split, acc, p, r, f1))
        #print(report)
        #print(confusion)
        return p, r, f1

    def run(self):
        ps, rs, f1s = [], [], []
        for split in range(self.config.total_splits):
            p, r, f1 = self.run_split(split)
            ps.append(p)
            rs.append(r)
            f1s.append(f1)
        ave_p = np.mean(ps)
        ave_r = np.mean(rs)
        ave_f1 = np.mean(f1s)
        print('average p {:.3f}, average r {:.3f}, average f1 {:.3f}'.format(ave_p, ave_r, ave_f1))
        result_path = 'result/{}_{}_{}.txt'.format(self.model_name, self.store.dataset, self.config.epochs)
        if not os.path.exists(result_path):
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write('p\tr\tf1\tconcat_feature\tuse_wiki\n')
        with open(result_path, 'a', encoding='utf-8') as f:
            f.write('{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(ave_p, ave_r, ave_f1, self.config.concat_feature, self.config.use_wiki))