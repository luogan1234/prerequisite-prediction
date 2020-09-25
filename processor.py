import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import json
import numpy as np
import tqdm
import random
from models.lstm_model import LSTM
from models.lstm_s_model import LSTM_S
from models.textcnn_model import TextCNN
from models.gcn_model import GCN

def name_to_model(config):
    if config.model == 'LSTM':
        return LSTM(config)
    if config.model == 'LSTM_S':
        return LSTM_S(config)
    if config.model == 'TextCNN':
        return TextCNN(config)
    if config.model == 'GCN':
        return GCN(config)
    raise NotImplementedError

class Processor:
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader
    
    def ce_loss(self, outputs, labels):
        labels = torch.tensor(labels, dtype=torch.long).to(self.config.device)
        loss = F.cross_entropy(outputs, labels)
        return loss
    
    def train_one_step(self, data):
        outputs = self.model(data)
        loss = self.ce_loss(outputs, data['labels'])
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
    
    def eval_one_step(self, data):
        with torch.no_grad():
            outputs = self.model(data)
            loss = self.ce_loss(outputs, data['labels'])
            outputs = outputs.cpu().numpy()
        return outputs, loss.item()
    
    def predict_one_step(self, data):
        with torch.no_grad():
            outputs = self.model(data)
            predicts = F.softmax(outputs, -1).cpu().numpy()
        return predicts
    
    def evaluate(self, data):
        self.model.eval()
        trues, preds = [], []
        eval_loss = 0
        for batch in data:
            trues.extend(batch['labels'])
            outputs, loss = self.eval_one_step(batch)
            preds.extend(np.argmax(outputs, axis=1).tolist())
            eval_loss += loss
        eval_loss /= len(data)
        self.model.train()
        acc = accuracy_score(trues, preds)
        p = precision_score(trues, preds, average='binary')
        r = recall_score(trues, preds, average='binary')
        f1 = f1_score(trues, preds, average='binary')
        return eval_loss, acc, p, r, f1
    
    def train(self):
        self.model = name_to_model(self.config)
        print('model parameters number: {}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        self.model.to(self.config.device)
        train, eval, test = self.data_loader.get_train()
        print('Batch data number: train {}, eval {}, test {}'.format(len(train), len(eval), len(test)))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        best_para = self.model.state_dict()
        min_loss = 1e16
        patience = 0
        train_tqdm = tqdm.tqdm(range(self.config.max_epochs))
        train_tqdm.set_description("Epoch %d | train_loss: %.3f eval_loss: %.3f" % (0, 0, 0))
        try:
            for epoch in train_tqdm:
                train_loss = 0.0
                for batch in train:
                    loss = self.train_one_step(batch)
                    train_loss += loss
                train_loss /= len(train)
                eval_loss, acc, p, r, f1 = self.evaluate(eval)
                train_tqdm.set_description("Epoch %d | train_loss: %.3f eval_loss: %.3f" % (epoch, train_loss, eval_loss))
                if eval_loss < min_loss:
                    patience = 0
                    min_loss = eval_loss
                    best_para = self.model.state_dict()
                patience += 1
                if patience > self.config.early_stop_time and epoch > self.config.min_check_epoch:
                    train_tqdm.close()
                    break
        except KeyboardInterrupt:
            train_tqdm.close()
            print('Exiting from training early, stop at epoch {}'.format(epoch))
        print('Train finished, stop at {} epochs, min eval_loss {:.3f}'.format(epoch, min_loss))
        test_loss, acc, p, r, f1 = self.evaluate(test)
        print('Test finished, test loss {:.3f}, acc {:.3f}, p {:.3f}, r {:.3f}, f1 {:.3f}'.format(test_loss, acc, p, r, f1))
        if self.config.save_model:
            with open('result/model_states/{}.pth'.format(self.config.model_name()), 'wb') as f:
                torch.save(best_para, f)
        if self.config.result_path is None:
            result_path = 'result/result/{}_{}.txt'.format(self.config.model, self.config.dataset)
        else:
            result_path = 'result/result/{}'.format(self.config.result_path)
        with open(result_path, 'a', encoding='utf-8') as f:
            obj = self.config.parameter_info()
            obj.update({'acc': round(acc, 3), 'p': round(p, 3), 'r': round(r, 3), 'f1': round(f1, 3)})
            f.write(json.dumps(obj)+'\n')
    
    def predict(self):
        match, data = self.data_loader.get_predict()
        n = len(self.data_loader.dataset.concepts)
        predicts = np.zeros((n, n, 2))
        tot = 0
        print('Start to predict, seed from 0 to 100...')
        for seed in range(100):
            self.config.seed = seed
            file = 'result/model_states/{}.pth'.format(self.config.model_name())
            if not os.path.exists(file):
                continue
            print('Seed {} result exists.'.format(seed))
            tot += 1
            self.model = name_to_model(self.config)
            with open(file, 'rb') as f:
                best_para = torch.load(f)
                self.model.load_state_dict(best_para)
            self.model.to(self.config.device)
            self.model.eval()
            for i, batch in tqdm.tqdm(enumerate(data)):
                batch_preds = self.predict_one_step(batch)
                i1 = batch['i1'].cpu().numpy().tolist()
                i2 = batch['i2'].cpu().numpy().tolist()
                for j in range(batch_preds.shape[0]):
                    predicts[i1[j]][i2[j]] += batch_preds[j]
        res = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    res.append({'i1': i, 'i2': j, 'label': match[i][j], 'predict': predicts[i][j]/tot})
        return res