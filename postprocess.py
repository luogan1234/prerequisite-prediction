import argparse
from data_loader import PreqDataLoader
from config import Config
from processor import Processor
import json
import os

def main():
    parser = argparse.ArgumentParser(description='Postprocess')
    parser.add_argument('-dataset', type=str, required=True, choices=['moocen', 'mooczh'])
    parser.add_argument('-model', type=str, required=True, choices=['lstm1', 'lstm2', 'gcn', 'gat'])
    parser.add_argument('-concat_user_feature', action='store_true')
    parser.add_argument('-embedding_dim', type=int, default=32)
    parser.add_argument('-encoding_dim', type=int, default=32)
    parser.add_argument('-info', type=str, default='')
    parser.add_argument('-cpu', action='store_true')
    args = parser.parse_args()
    config = Config(args.dataset, args.model, args.concat_user_feature, args.embedding_dim, args.encoding_dim, args.info, 0, args.cpu)
    data_loader = PreqDataLoader(config)
    processor = Processor(config, data_loader)
    res = processor.predict()
    with open('result/predictions/{}_{}.json'.format(args.dataset, args.model), 'w', encoding='utf-8') as f:
        for pred in res:
            obj = {'c1': data_loader.dataset.concepts[pred['i1']], 'c2': data_loader.dataset.concepts[pred['i2']], 'label': pred['label'], 'prediction': pred['predict'].tolist()}
            f.write(json.dumps(obj, ensure_ascii=False)+'\n')

if __name__ == '__main__':
    main()