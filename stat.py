import numpy as np
import os

def eval():
    files = os.listdir('result/')
    for file in files:
        if 'mooczh' not in file or 'GCN' not in file:
            continue
        ps, rs, f1s = [], [], []
        with open('result/'+file, 'r', encoding='utf-8') as f:
            data = f.read().split('\n')[1:]
            for line in data:
                if line:
                    items = line.split('\t')[:3]
                    ps.append(float(items[0]))
                    rs.append(float(items[1]))
                    f1s.append(float(items[2]))
        print(file)
        print('p average: {:.3f}, r average: {:.3f}, f1 average: {:.3f}\n'.format(np.mean(ps), np.mean(rs), np.mean(f1s)))

if __name__ == '__main__':
    eval()