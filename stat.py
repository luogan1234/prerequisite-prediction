import numpy as np
import os
import json

def stat():
    with open('result/result.txt', 'r') as f:
        res = []
        for line in f:
            obj = json.loads(line)
            res.append(obj)
    dataset_union = set([obj['dataset'] for obj in res])
    model_union = set([obj['model'] for obj in res])
    concat_user_feature_union = set([obj['concat_user_feature'] for obj in res])
    embedding_dim_union = set([obj['embedding_dim'] for obj in res])
    encoding_dim_union = set([obj['encoding_dim'] for obj in res])
    info_union = set([obj['info'] for obj in res])
    seed_union = set([obj['seed'] for obj in res])
    for dataset in dataset_union:
        for model in model_union:
            for concat_user_feature in concat_user_feature_union:
                for embedding_dim in embedding_dim_union:
                    for encoding_dim in encoding_dim_union:
                        for info in info_union:
                            subset = []
                            for seed in seed_union:
                                conditions = {'dataset': dataset, 'model': model, 'concat_user_feature': concat_user_feature, 'embedding_dim': embedding_dim, 'encoding_dim': encoding_dim, 'info': info, 'seed': seed}
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