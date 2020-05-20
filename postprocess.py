import argparse
from data_loader import DataLoader
from config import Config
from processor import Processor
import json
import os

def main():
    parser = argparse.ArgumentParser(description='Postprocess')
    parser.add_argument('-model', type=str, required=True, choices=['LSTM', 'TextCNN', 'GCN'])
    parser.add_argument('-dataset', type=str, required=True, choices=['moocen', 'mooczh'])
    parser.add_argument('-max_term_length', type=int, default=7)
    parser.add_argument('-feature_dim', type=int, default=24)
    args = parser.parse_args()
    if args.dataset in ['moocen']:
        lang = 'en'
    if args.dataset in ['mooczh']:
        lang = 'zh'
    if not os.path.exists('predictions/'):
        os.mkdir('predictions/')
    data_loader = DataLoader(args.dataset, args.model, lang, args.max_term_length)
    config = Config(data_loader, args.feature_dim)
    processor = Processor(args.model, data_loader, config)
    predicts = processor.predict()
    if predicts:
        with open('predictions/{}_{}.json'.format(args.model, args.dataset), 'w', encoding='utf-8') as f:
            for pred in predicts:
                obj = {'c1': data_loader.concepts[pred['input'][0]], 'c2': data_loader.concepts[pred['input'][1]], 'label': pred['label'], 'predict': pred['predict'].tolist()}
                f.write(json.dumps(obj, ensure_ascii=False)+'\n')

if __name__ == '__main__':
    main()