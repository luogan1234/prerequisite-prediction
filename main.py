import argparse
from data_handler import DataHandler
from processor import Processor
from config import Config
import pickle
import os

def main():
    parser = argparse.ArgumentParser(description='Prerequisite prediction')
    parser.add_argument('-model', type=str, required=True, choices=['LSTM', 'TextCNN', 'GCN', 'GCN_LSTM', 'MLP'], help='LSTM | TextCNN | GCN | GCN_LSTM | MLP')
    parser.add_argument('-dataset', type=str, required=True, choices=['mooczh', 'moocen'], help='mooczh | moocen')
    parser.add_argument('-max_term_length', type=int, default=7)
    parser.add_argument('-max_sentence_length', type=int, default=100)
    parser.add_argument('-use_wiki', action='store_true')
    parser.add_argument('-use_cpu', action='store_true')
    parser.add_argument('-embedding_dim', type=int, default=16)
    parser.add_argument('-mlp_dim', type=int, default=24)
    parser.add_argument('-epochs', type=int, default=500)
    args = parser.parse_args()
    if args.dataset in ['moocen']:
        lang = 'en'
    if args.dataset in ['mooczh']:
        lang = 'zh'
    if not os.path.exists('tmp/'):
        os.path.mkdir('tmp/')
    store_path = 'tmp/{}_{}_{}_{}_{}.pkl'.format(args.dataset, lang, args.use_wiki, args.max_term_length, args.max_sentence_length)
    if os.path.exists(store_path):
        with open(store_path, 'rb') as f:
            store = pickle.load(f)
        store.model_name = args.model
    else:
        store = DataHandler(args.dataset, args.model, lang, args.use_wiki, args.max_term_length, args.max_sentence_length)
        with open(store_path, 'wb') as f:
            pickle.dump(store, f)
    config = Config(store, args.embedding_dim, args.mlp_dim)
    config.epochs = args.epochs
    if args.use_cpu:
        config.use_gpu = False
    processor = Processor(args.model, store, config)
    processor.run(True)

if __name__ == '__main__':
    main()