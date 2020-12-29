import numpy as np
import os
import json

def stat():
    with open('result/result.txt', 'r') as f:
        res = []
        for line in f:
            obj = json.loads(line)
            res.append(obj)
    dataset_union = list(set([obj['dataset'] for obj in res]))
    model_union = list(set([obj['model'] for obj in res]))
    concat_feature_union = list(set([obj['concat_feature'] for obj in res]))
    embedding_dim_union = list(set([obj['embedding_dim'] for obj in res]))
    encoding_dim_union = list(set([obj['encoding_dim'] for obj in res]))
    info_union = list(set([obj['info'] for obj in res]))
    seed_union = list(set([obj['seed'] for obj in res]))
    dataset_union.sort()
    model_union.sort()
    concat_feature_union.sort()
    embedding_dim_union.sort()
    encoding_dim_union.sort()
    info_union.sort()
    seed_union.sort()
    for dataset in dataset_union:
        for model in model_union:
            for concat_feature in concat_feature_union:
                for embedding_dim in embedding_dim_union:
                    for encoding_dim in encoding_dim_union:
                        for info in info_union:
                            subset = []
                            for seed in seed_union:
                                conditions = {'dataset': dataset, 'model': model, 'concat_feature': concat_feature, 'embedding_dim': embedding_dim, 'encoding_dim': encoding_dim, 'info': info, 'seed': seed}
                                target = None
                                for obj in res:
                                    if all([obj[key] == conditions[key] for key in conditions]):
                                        target = obj
                                if target is not None:
                                    subset.append(target)
                            if subset:
                                acc, p, r, f1 = [], [], [], []
                                for obj in subset:
                                    acc.append(obj['acc'])
                                    p.append(obj['p'])
                                    r.append(obj['r'])
                                    f1.append(obj['f1'])
                                conditions.pop('seed')
                                print(conditions)
                                print('average acc: {:.3f}, p: {:.3f}, r: {:.3f}, f1: {:.3f}\n'.format(np.mean(acc), np.mean(p), np.mean(r), np.mean(f1)))

if __name__ == '__main__':
    stat()