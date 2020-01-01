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

class Processor(object):
    def __init__(self, model, store, config):
        self.model = model
        self.store = store
        self.config = config
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
    
    def train_one_step(self, inputs, labels):
        labels = self.to_var(labels)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def eval_one_step(self, inputs, labels):
        labels = self.to_var(labels)
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
        inputs = [self.to_var(np.array(input)) for input in inputs]
        labels = np.array(labels, dtype=np.int64)
        return inputs, labels

    def run(self):
        if self.config.use_gpu:
            self.model.cuda()
        for split in range(self.config.total_splits):
            print("Split {} starts:".format(split))
            train, eval = self.store.prepare_data(split, self.config.total_splits)
            train_steps = (len(train) - 1) // self.config.batch_size + 1
            eval_steps = (len(eval) - 1) // self.config.batch_size + 1
            training_range = tqdm.tqdm(range(self.config.epochs))
            training_range.set_description("Epoch %d | loss: %.3f" % (0, 0))
            for epoch in training_range:
                res = 0.0
                random.shuffle(train)
                for i in range(train_steps):
                    inputs, labels = self.get_batch_data(train[i*self.config.batch_size: (i+1)*self.config.batch_size])
                    loss = self.train_one_step(inputs, labels)
                    res += loss
                training_range.set_description("Epoch %d | loss: %.3f" % (epoch, res))
            if self.config.save_dir:
                save_path = os.path.join(self.config.save_dir, '{}_{}.ckpt'.format(self.model.name, split))
                print("Split {} finished, save at {}".format(split, self.config.save_dir))
                torch.save(self.model.state_dict(), save_path)
            labels_all = np.array([], dtype=int)
            predicts_all = np.array([], dtype=int)
            for i in range(eval_steps):
                inputs, labels = self.get_batch_data(eval[i*self.config.batch_size: (i+1)*self.config.batch_size])
                predicts, loss = self.eval_one_step(inputs, labels)
                labels_all = np.append(labels_all, labels)
                predicts_all = np.append(predicts_all, predicts)
            acc = metrics.accuracy_score(labels_all, predicts_all)
            report = metrics.classification_report(labels_all, predicts_all)
            confusion = metrics.confusion_matrix(labels_all, predicts_all)
            print('Split acc: {}'.format(acc))
            print(report)
            print(confusion)

    def to_var(self, x):
        if self.config.use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))