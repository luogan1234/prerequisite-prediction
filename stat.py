import numpy as np
import os

def eval():
    files = os.listdir('result/')
    files.sort()
    for file in files:
        ps, rs, f1s = [], [], []
        with open('result/'+file, 'r', encoding='utf-8') as f:
            data = f.read().split('\n')[1:]
            for line in data:
                if line:
                    p, r, f1 = line.split('\t')[:3]
                    p, r, f1 = float(p), float(r), float(f1)
                    if p < 0.5 or r < 0.5 or f1 < 0.5:
                        continue
                    ps.append(p)
                    rs.append(r)
                    f1s.append(f1)
        print(file)
        print('p average: {:.3f}, r average: {:.3f}, f1 average: {:.3f}\n'.format(np.mean(ps), np.mean(rs), np.mean(f1s)))

if __name__ == '__main__':
    eval()