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
    parser = argparse.ArgumentParser(description='Prerequisite prediction')
    parser.add_argument('-dataset', type=str, required=True, choices=['moocen', 'mooczh'])
    parser.add_argument('-model', type=str, required=True, choices=['LSTM', 'LSTM_S', 'TextCNN', 'GCN'])
    parser.add_argument('-feature_dim', type=int, default=24)
    parser.add_argument('-result_path', type=str, default=None)
    parser.add_argument('-save_model', action='store_true')
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-cpu', action='store_true')
    args = parser.parse_args()
    set_seed(args.seed)
    config = Config(args.dataset, args.model, args.feature_dim, args.result_path, args.save_model, args.seed, args.cpu)
    data_loader = PreqDataLoader(config)
    processor = Processor(config, data_loader)
    processor.train()

if __name__ == '__main__':
    main()