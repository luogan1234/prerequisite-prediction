import argparse
from data_handler import DataHandler
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
    parser.add_argument('-model', type=str, required=True, choices=['LSTM', 'TextCNN', 'GCN'], help='LSTM | TextCNN | GCN')
    parser.add_argument('-dataset', type=str, required=True, choices=['mooczh', 'moocen'], help='mooczh | moocen')
    parser.add_argument('-max_term_length', type=int, default=7)
    parser.add_argument('-max_sentence_length', type=int, default=100)
    parser.add_argument('-use_cpu', action='store_true')
    parser.add_argument('-embedding_dim', type=int, default=36)
    parser.add_argument('-feature_dim', type=int, default=36)
    parser.add_argument('-max_epochs', type=int, default=500)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-graph', type=str, default='graph.npy')
    parser.add_argument('-output', type=str, default=None)
    args = parser.parse_args()
    set_seed(args.seed)
    if args.dataset in ['moocen']:
        lang = 'en'
    if args.dataset in ['mooczh']:
        lang = 'zh'
    if not os.path.exists('result/'):
        os.mkdir('result/')
    store = DataHandler(args.dataset, args.model, lang, args.max_term_length, args.max_sentence_length, args.graph)
    config = Config(store, args.embedding_dim, args.feature_dim, args.max_epochs, args.use_cpu)
    processor = Processor(args.model, store, config)
    processor.run(args.output)

if __name__ == '__main__':
    main()