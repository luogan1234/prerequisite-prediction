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
from model.model import Model

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
        scores = {'acc': round(acc, 3), 'p': round(p, 3), 'r': round(r, 3), 'f1': round(f1, 3)}
        return eval_loss, scores
    
    def train(self):
        self.model = Model(self.config)
        print('model parameters number: {}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        self.model.to(self.config.device)
        train, eval, test = self.data_loader.get_train()
        print('Batch data number ({} samples for train batch, {} samples for eval batch): train {}, eval {}, test {}'.format(self.config.batch_size('train'), self.config.batch_size('eval'), len(train), len(eval), len(test)))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        best_para = self.model.state_dict()
        min_loss = 1e16
        patience = 0
        train_tqdm = tqdm.tqdm(range(self.config.max_epochs))
        train_tqdm.set_description('Epoch {} | train_loss: {:.3f} eval_loss: {:.3f}'.format(0, 0, 0))
        try:
            for epoch in train_tqdm:
                train_loss = 0.0
                for batch in train:
                    loss = self.train_one_step(batch)
                    train_loss += loss
                train_loss /= len(train)
                eval_loss, scores = self.evaluate(eval)
                train_tqdm.set_description('Epoch {} | train_loss: {:.3f} eval_loss: {:.3f}'.format(epoch, train_loss, eval_loss))
                if eval_loss < min_loss:
                    patience = 0
                    min_loss = eval_loss
                    best_para = self.model.state_dict()
                patience += 1
                if patience > self.config.early_stop_time:
                    train_tqdm.close()
                    break
        except KeyboardInterrupt:
            train_tqdm.close()
            print('Exiting from training early, stop at epoch {}'.format(epoch))
        print('Train finished, stop at {} epochs, min eval_loss {:.3f}'.format(epoch, min_loss))
        test_loss, scores = self.evaluate(test)
        print('Test finished, test loss {:.3f},'.format(test_loss), scores)
        with open('result/model_states/{}.pth'.format(self.config.store_name()), 'wb') as f:
            torch.save(best_para, f)
        with open('result/result.txt', 'a', encoding='utf-8') as f:
            obj = self.config.parameter_info()
            obj.update(scores)
            f.write(json.dumps(obj)+'\n')
    
    def predict(self):
        match, data = self.data_loader.get_predict()
        n = len(self.data_loader.dataset.concepts)
        predicts = np.zeros((n, n, 2))
        tot = 0
        print('Start to predict, seed from 0 to 100...')
        for seed in range(100):
            self.config.seed = seed
            file = 'result/model_states/{}.pth'.format(self.config.store_name())
            if not os.path.exists(file):
                continue
            tot += 1
            self.model = Model(self.config)
            with open(file, 'rb') as f:
                best_para = torch.load(f)
                self.model.load_state_dict(best_para)
            self.model.to(self.config.device)
            self.model.eval()
            for i, batch in enumerate(data):
                batch_preds = self.predict_one_step(batch)
                i1 = batch['i1'].cpu().numpy().tolist()
                i2 = batch['i2'].cpu().numpy().tolist()
                for j in range(batch_preds.shape[0]):
                    predicts[i1[j]][i2[j]] += batch_preds[j]
            print('Model with seed {} done.'.format(seed))
        res = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    res.append({'i1': i, 'i2': j, 'label': match[i][j], 'predict': predicts[i][j]/tot})
        return res