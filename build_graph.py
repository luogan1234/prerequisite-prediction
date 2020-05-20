import numpy as np
import argparse
import json
from collections import Counter
import math
import random
import tqdm

def add_edge(graph, c1s, c2s, d):
    if c1s and c2s:
        d /= max(len(c1s), len(c2s))
        for c1 in c1s:
            for c2 in c2s:
                if c1 != c2:
                    graph[c1][c2] += d

def build_concept_graph(dataset, alpha, video_order, course_dependency, user_act, user_prop, user_num, user_act_type):
    labels = [video_order, course_dependency, user_act]
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
    video_to_concept, course_to_concept, course_to_video, video_to_course = {}, {}, {}, {}
    with open(prefix+'video-concepts.json', 'r', encoding='utf-8') as f:
        for line in f.read().split('\n'):
            if line:
                obj = json.loads(line)
                v = videos.index(obj['video'])
                cs = [concepts.index(c) for c in obj['concepts']]
                video_to_concept[v] = cs
    pre = {}
    for v in range(vn):
        pre[v] = []
    cgraph, vgraph = np.zeros((3, cn, cn)), np.zeros((3, vn, vn))
    with open(prefix+'course-videos.json', 'r', encoding='utf-8') as f:
        for line in f.read().split('\n'):
            if line:
                obj = json.loads(line)
                c = obj['course']
                course_to_concept[c] = set()
                for v in obj['videos']:
                    course_to_concept[c] = course_to_concept[c].union(video_to_concept.get(videos.index(v), []))
                vs = [videos.index(v) for v in obj['videos']]
                course_to_video[c] = vs
                for v in vs:
                    video_to_course[v] = c
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
                    add_edge(cgraph[1], course_to_concept[c1], course_to_concept[c2], 1)
    if user_act:
        with open(prefix+'user-videos.json', 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f.read().split('\n') if line]
            if user_num >= 0:
                w = min(len(data), user_num)
            else:
                w = min(len(data), int(user_prop*len(data)))
            data = random.sample(data, w)
            if data:
                w = len(data)
                w = math.log(w) / w
                tot = [0]*4
                for obj in data:
                    vs = [videos.index(v) for v in obj['videos']]
                    vs_set = set(vs)
                    n = len(vs)
                    if user_act_type[0]:  # sequential
                        for i in range(n):
                            for j in range(i+1, n):
                                if vs[i] in pre[vs[j]]:
                                    s = alpha ** (j-i) * w
                                    vgraph[2][vs[i]][vs[j]] += s
                                    tot[0] += s
                    if user_act_type[1]:  # cross_course
                        last_video_place = {}
                        for i in range(n):
                            ci = video_to_course[vs[i]]
                            if ci not in last_video_place:
                                last_video_place[ci] = 0
                            last_video_place[ci] = i
                        for i in range(n):
                            ci = video_to_course[vs[i]]
                            for j in range(i+1, last_video_place[ci]):
                                cj = video_to_course[vs[j]]
                                if ci != cj:
                                    s = alpha ** (j-i) * w
                                    vgraph[2][vs[j]][vs[i]] += s
                                    tot[1] += s
                    if user_act_type[2]:  # backward
                        for i in range(n):
                            for j in range(i+1, n):
                                if vs[j] in pre[vs[i]]:
                                    s = alpha ** (j-i) * w
                                    vgraph[2][vs[j]][vs[i]] += s
                                    tot[2] += s
                    if user_act_type[3]:  # skip
                        for v in vs:
                            for i, pv in enumerate(pre[v]):
                                if pv not in vs_set:
                                    s = - alpha ** (i+1) * w
                                    vgraph[2][pv][v] += s
                                    tot[3] += s
                print('(video graph) sequential w: {:.3f}, cross_course w: {:.3f}, backward w: {:.3f}, skip w: {:.3f}'.format(tot[0], tot[1], tot[2], tot[3]))
    graphs = []
    for k in range(3):
        if labels[k]:
            for i in range(vn):
                for j in range(vn):
                    if vgraph[k][i][j] > 0:
                        add_edge(cgraph[k], video_to_concept[i], video_to_concept[j], vgraph[k][i][j])
            print('(video graph) graph {}, covered edge proportion: {:.3f}, total edge weight: {:.3f}'.format(k, len(vgraph[k][vgraph[k]>0]) / (vn*vn), np.sum(vgraph[k][vgraph[k]>0])))
            print('(concept graph) graph {}, covered edge proportion: {:.3f}, total edge weight: {:.3f}'.format(k, len(cgraph[k][cgraph[k]>0]) / (cn*cn), np.sum(cgraph[k])))
            graphs.append(cgraph[k])
    np.save(prefix+'graph.npy', np.array(graphs))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description='Prerequisite prediction')
    parser.add_argument('-dataset', type=str, required=True, choices=['moocen', 'mooczh'])
    parser.add_argument('-alpha', type=float, default=None)
    parser.add_argument('-no_video_order', action='store_true')
    parser.add_argument('-no_course_dependency', action='store_true')
    parser.add_argument('-no_user_act', action='store_true')
    parser.add_argument('-user_prop', type=float, default=1.0)
    parser.add_argument('-user_num', type=int, default=-1)
    parser.add_argument('-user_act_type', type=str, default='all', choices=['all', 'none', 'sequential_only', 'cross_course_only', 'backward_only', 'skip_only', 'no_sequential', 'no_cross_course', 'no_backward', 'no_skip'])
    parser.add_argument('-seed', type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    if not args.alpha:
        if args.dataset == 'moocen':
            args.alpha = 0.1
        if args.dataset == 'mooczh':
            args.alpha = 0.3
    assert 0.0 <= args.alpha <= 1.0 and 0.0 <= args.user_prop <= 1.0 and args.user_num >= -1
    video_order = not(args.no_video_order)
    course_dependency = not(args.no_course_dependency) if args.dataset in ['mooczh'] else False
    user_act = not(args.no_user_act) if args.dataset in ['mooczh'] else False
    if args.user_act_type in ['all', 'none']:
        user_act_type = [True]*4 if args.user_act_type == 'all' else [False]*4
    elif args.user_act_type.endswith('only'):
        user_act_type = [False]*4
        p = ['sequential_only', 'cross_course_only', 'backward_only', 'skip_only'].index(args.user_act_type)
        user_act_type[p] = True
    else:
        user_act_type = [True]*4
        p = ['no_sequential', 'no_cross_course', 'no_backward', 'no_skip'].index(args.user_act_type)
        user_act_type[p] = False
    build_concept_graph(args.dataset, args.alpha, video_order, course_dependency, user_act, args.user_prop, args.user_num, user_act_type)

if __name__ == '__main__':
    main()
   