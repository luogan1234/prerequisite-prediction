import os

if not os.path.exists('data/'):
    os.system('wget http://lfs.aminer.cn/misc/moocdata/data/prerequisite_dataset.zip')
    os.system('unzip prerequisite_dataset.zip')
if not os.path.exists('result/'):
    os.mkdir('result/')
if not os.path.exists('result/result/'):
    os.mkdir('result/result/')
if not os.path.exists('result/model_states/'):
    os.mkdir('result/model_states/')
if not os.path.exists('result/predictions/'):
    os.mkdir('result/predictions/')