import argparse
from config import Config
from data_loader import PreqDataLoader
from processor import Processor
import pickle
import os
import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

def main():
    if not os.path.exists('result/'):
        os.mkdir('result/')
    if not os.path.exists('result/model_states/'):
        os.mkdir('result/model_states/')
    if not os.path.exists('result/predictions/'):
        os.mkdir('result/predictions/')
    parser = argparse.ArgumentParser(description='Prerequisite prediction')
    parser.add_argument('-dataset', type=str, required=True, choices=['moocen', 'mooczh'])
    parser.add_argument('-model', type=str, required=True, choices=['lstm', 'gcn', 'gat'])
    parser.add_argument('-concat_user_feature', action='store_true')
    parser.add_argument('-embedding_dim', type=int, default=32)
    parser.add_argument('-encoding_dim', type=int, default=32)
    parser.add_argument('-info', type=str, default='')
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-cpu', action='store_true')
    args = parser.parse_args()
    set_seed(args.seed)
    config = Config(args.dataset, args.model, args.concat_user_feature, args.embedding_dim, args.encoding_dim, args.info, args.seed, args.cpu)
    data_loader = PreqDataLoader(config)
    processor = Processor(config, data_loader)
    processor.train()

if __name__ == '__main__':
    main()