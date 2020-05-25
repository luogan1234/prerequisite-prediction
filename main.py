import argparse
from data_loader import DataLoader
from processor import Processor
from config import Config
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
    parser.add_argument('-model', type=str, required=True, choices=['LSTM', 'LSTM_S', 'LSTM_GCN', 'TextCNN', 'GCN'])
    parser.add_argument('-dataset', type=str, required=True, choices=['moocen', 'mooczh'])
    parser.add_argument('-max_term_length', type=int, default=7)
    parser.add_argument('-feature_dim', type=int, default=24)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-result_path', type=str, default=None)
    parser.add_argument('-output_model', action='store_true')
    args = parser.parse_args()
    set_seed(args.seed)
    if args.dataset in ['moocen']:
        lang = 'en'
    if args.dataset in ['mooczh']:
        lang = 'zh'
    if not os.path.exists('model_states/'):
        os.mkdir('model_states/')
    if not os.path.exists('result/'):
        os.mkdir('result/')
    data_loader = DataLoader(args.dataset, args.model, lang, args.max_term_length)
    config = Config(data_loader, args.feature_dim)
    processor = Processor(args.model, data_loader, config)
    processor.run(args.result_path, args.output_model)

if __name__ == '__main__':
    main()