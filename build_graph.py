import numpy as np
import argparse
import json
from collections import Counter
import math

def add_edge(graph, c1s, c2s, d):
    if c1s and c2s:
        d /= max(len(c1s), len(c2s))
        for c1 in c1s:
            for c2 in c2s:
                if c1 != c2:
                    graph[c1][c2] += d

def build_concept_graph(dataset, alpha, video_order, course_dependency, user_data):
    labels = [video_order, course_dependency, user_data]
    prefix = 'dataset/{}/'.format(dataset)
    concepts = []
    with open(prefix+'concepts.txt', 'r', encoding='utf-8') as f:
        for line in f.read().split('\n'):
            if line:
                concepts.append(line)
    videos = []
    with open(prefix+'captions.txt', 'r', encoding='utf-8') as f:
        for line in f.read().split('\n'):
            if line:
                videos.append(line.split(',')[0])
    cn, vn = len(concepts), len(videos)
    v2c = {}
    with open(prefix+'video-concepts.json', 'r', encoding='utf-8') as f:
        for line in f.read().split('\n'):
            if line:
                obj = json.loads(line)
                v = videos.index(obj['video'])
                cs = [concepts.index(c) for c in obj['concepts']]
                v2c[v] = cs
    pre = {}
    for v in range(vn):
        pre[v] = []
    c2c = {}
    cgraph, vgraph = np.zeros((3, cn, cn)), np.zeros((3, vn, vn))
    with open(prefix+'course-videos.json', 'r', encoding='utf-8') as f:
        for line in f.read().split('\n'):
            if line:
                obj = json.loads(line)
                c = obj['course']
                c2c[c] = set()
                for v in obj['videos']:
                    c2c[c] = c2c[c].union(v2c.get(videos.index(v), []))
                vs = [videos.index(v) for v in obj['videos']]
                n = len(vs)
                for i in range(n):
                    if video_order:
                        for j in range(i+1, n):
                            vgraph[0][vs[i]][vs[j]] += alpha ** (j-i)
                for i in range(n):
                    for j in range(i-1, -1, -1):
                        pre[vs[i]].append(vs[j])
    if course_dependency:
        with open(prefix+'course_dependency.txt', 'r', encoding='utf-8') as f:
            for line in f.read().split('\n'):
                if line:
                    c1, c2 = line.split('\t')
                    add_edge(cgraph[1], c2c[c1], c2c[c2], 1)
    if user_data:
        with open(prefix+'user-videos.json', 'r', encoding='utf-8') as f:
            data = [line for line in f.read().split('\n') if line]
            w = len(data)
            w = math.log(w) / w
            for line in data:
                obj = json.loads(line)
                vs = [videos.index(v) for v in obj['videos']]
                vs_set = set(vs)
                n = len(vs)
                for i in range(n):
                    for j in range(i+1, n):
                        vgraph[2][vs[i]][vs[j]] += alpha ** (j-i) * w
                for v in vs:
                    for i, pv in enumerate(pre[v]):
                        if pv not in vs_set:
                            vgraph[2][pv][v] -= alpha ** (i+1) * w
    graphs = []
    for k in range(3):
        if labels[k]:
            for i in range(vn):
                for j in range(vn):
                    if vgraph[k][i][j] > 0:
                        add_edge(cgraph[k], v2c[i], v2c[j], vgraph[k][i][j])
            print('graph {}, concept graph edge proportion: {:.3f}'.format(k, len(cgraph[k][cgraph[k]>0]) / (cn*cn)))
            graphs.append(cgraph[k])
    np.save(prefix+'graph.npy', np.array(graphs))

def main():
    parser = argparse.ArgumentParser(description='Prerequisite prediction')
    parser.add_argument('-dataset', type=str, required=True, choices=['mooczh', 'moocen'], help='mooczh | moocen')
    parser.add_argument('-alpha', type=float, default=0.3)
    parser.add_argument('-no_video_order', action='store_true')
    parser.add_argument('-no_course_dependency', action='store_true')
    parser.add_argument('-no_user_data', action='store_true')
    args = parser.parse_args()
    video_order = not(args.no_video_order)
    course_dependency = not(args.no_course_dependency) if args.dataset in ['mooczh'] else False
    user_data = not(args.no_user_data) if args.dataset in ['mooczh'] else False
    build_concept_graph(args.dataset, args.alpha, video_order, course_dependency, user_data)

if __name__ == '__main__':
    main()
   