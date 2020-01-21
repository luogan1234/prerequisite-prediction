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

def name_to_model(name, config):
    if name == 'LSTM':
        return LSTM(config)
    if name == 'TextCNN':
        return TextCNN(config)
    if name == 'GCN':
        return GCN(config)
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
    
    def evaluate(self, data):
        self.model.eval()
        labels_all = np.array([], dtype=int)
        predicts_all = np.array([], dtype=int)
        eval_loss = 0
        eval_steps = (len(data) - 1) // self.config.batch_size + 1
        for i in range(eval_steps):
            inputs, labels = self.get_batch_data(data[i*self.config.batch_size: (i+1)*self.config.batch_size])
            predicts, loss = self.eval_one_step(inputs, labels)
            labels_all = np.append(labels_all, labels)
            predicts_all = np.append(predicts_all, predicts)
            eval_loss += loss
        eval_loss /= eval_steps
        self.model.train()
        acc = metrics.accuracy_score(labels_all, predicts_all)
        p = metrics.precision_score(labels_all, predicts_all, average='binary')
        r = metrics.recall_score(labels_all, predicts_all, average='binary')
        f1 = metrics.f1_score(labels_all, predicts_all, average='binary')
        return eval_loss, acc, p, r, f1
    
    def run_split(self, split):
        print("Split {} starts, use model {}".format(split, self.model_name))
        self.model = name_to_model(self.model_name, self.config)
        if self.config.use_gpu:
            self.model.cuda()
        best_para = self.model.state_dict()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        train, eval, test = self.store.prepare_data(split, self.config.total_splits)
        train_steps = (len(train) - 1) // self.config.batch_size + 1
        training_tqdm = tqdm.tqdm(range(self.config.max_epochs))
        training_tqdm.set_description("Epoch %d | train_loss: %.3f eval_loss: %.3f" % (0, 0, 0))
        min_eval_loss = 1e16
        patience = 0
        for epoch in training_tqdm:
            train_loss = 0.0
            random.shuffle(train)
            for i in range(train_steps):
                inputs, labels = self.get_batch_data(train[i*self.config.batch_size: (i+1)*self.config.batch_size])
                loss = self.train_one_step(inputs, labels)
                train_loss += loss
            train_loss /= train_steps
            eval_loss, acc, p, r, f1 = self.evaluate(eval)
            training_tqdm.set_description("Epoch %d | train_loss: %.3f eval_loss: %.3f" % (epoch, train_loss, eval_loss))
            if eval_loss < min_eval_loss:
                patience = 0
                min_eval_loss = eval_loss
                best_para = self.model.state_dict()
            patience += 1
            if patience > self.config.early_stop_time and epoch > self.config.min_check_epoch:
                training_tqdm.close()
                break
        print('Split {} train finished, min eval loss {:.3f}, stop at {} epochs'.format(split, min_eval_loss, epoch))
        self.model.load_state_dict(best_para)
        test_loss, acc, p, r, f1 = self.evaluate(test)
        print('Split {} test finished, test loss {:.3f} acc {:.3f} p {:.3f} r {:.3f} f1 {:.3f}'.format(split, test_loss, acc, p, r, f1))
        return p, r, f1

    def run(self, output):
        model = name_to_model(self.model_name, self.config)
        print('model parameters number: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        ps, rs, f1s = [], [], []
        for split in range(self.config.total_splits):
            p, r, f1 = self.run_split(split)
            ps.append(p)
            rs.append(r)
            f1s.append(f1)
            if not output:
                result_path = 'result/{}_{}_{}_{}.txt'.format(self.model_name, self.store.dataset, self.config.embedding_dim, self.config.feature_dim)
            else:
                result_path = 'result/{}'.format(output)
            if not os.path.exists(result_path):
                with open(result_path, 'w', encoding='utf-8') as f:
                    f.write('p\tr\tf1\tsplit\n')
            with open(result_path, 'a', encoding='utf-8') as f:
                f.write('{:.3f}\t{:.3f}\t{:.3f}\t{}\n'.format(p, r, f1, split))
        ave_p = np.mean(ps)
        ave_r = np.mean(rs)
        ave_f1 = np.mean(f1s)
        print('average p {:.3f}, average r {:.3f}, average f1 {:.3f}'.format(ave_p, ave_r, ave_f1))
        