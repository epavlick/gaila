"""
Script for generating nearest neighbors/cluster analyses for GAILA Milestone 4 report.
"""
import os
import sys
import csv
import json
import gzip
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import log
from collections import Counter
from nltk.util import ngrams
import pandas as pd
import plotly.express as px
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import argparse
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def make_heatmap(M, row_names, col_names, row_tots, col_tots,
                 col_cutoff = 20, row_cutoff = 20, row_scale=0.15, col_scale=0.2, cmap='coolwarm',
                sort = None, annotate=False, saveas=None):
    
    if (row_cutoff is not None) and (col_cutoff is not None):          
        plot_M = M[row_tots>row_cutoff, :][:, col_tots>col_cutoff]
        plot_row = [w for i, w in enumerate(row_names) if row_tots[i] > row_cutoff]
        plot_col = [w for i, w in enumerate(col_names) if col_tots[i] > col_cutoff]
    else:
        plot_M = M
        plot_row = row_names
        plot_col = col_names
        
    if sort == 'diag':
        idxs = np.argmax(plot_M, axis=1) 
        order = np.argsort(idxs)
        sorted_M = plot_M[order]
        plot_M = sorted_M
        plot_row = [plot_row[i] for i in order]
    elif sort == 'strength':
        idxs = np.max(plot_M, axis=1) 
        order = np.flip(np.argsort(idxs))
        sorted_M = plot_M[order]
        plot_M = sorted_M
        plot_row = [plot_row[i] for i in order]
    elif sort == 'row':
        rows = sorted(list(set(plot_row)))
        order = np.argsort([rows.index(n) for n in plot_row])
        sorted_M = plot_M[order]
        plot_M = sorted_M
        plot_row = [plot_row[i] for i in order]
        
    height, width = plot_M.shape
    plt.figure(figsize=(int(round(width*col_scale)), int(round(height*row_scale))))

    if annotate:
        sns.heatmap(plot_M, cmap=cmap, annot=True, fmt=".01f")
    else:
        sns.heatmap(plot_M, cmap=cmap, cbar=False)
    plt.yticks(np.arange(plot_M.shape[0])+0.5, plot_row, rotation=0)
    plt.xticks(np.arange(plot_M.shape[1])+0.5, plot_col, rotation=90)
    if saveas:
        plt.savefig(saveas, bbox_inches='tight')
    plt.show()
    plt.clf()
    
def get_pmi(freq):
    col_tots = np.sum(freq, axis=0)
    row_tots = np.sum(freq, axis=1)
    tot = np.sum(freq)
    
    rows, cols = freq.shape
    PMI = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            pmi = (freq[i][j]/tot) / ((row_tots[i]/tot) * (col_tots[j]/tot))
            PMI[i][j] = log(pmi) if pmi > 0 else 0
    return PMI
    
def make_mats(D, POS = 'VERB', window_size=0):
    poslst = POS.split(',')
    zs = sorted(list(set([r['z'] for r in data])),
                key=lambda e:int(e) if not e == 'NA' else -1)
    words = sorted(list(set([r['lemma'] for r in D if r['pos'] in poslst])))
  
    
    N = len(D)
    freq = np.zeros((len(words), len(zs)))
    straight_count = np.zeros((len(words), len(zs)))
    for i, d in enumerate(D):
        if d['pos'] in poslst:
            w = d['lemma']
            for j in range(max(0, i-window_size), min(i+window_size, N)+1):
                s = D[j]['z']
                freq[words.index(w)][zs.index(s)] += 1
                if i == j:
                    straight_count[words.index(w)][zs.index(s)] += 1
    
    col_tots = np.sum(straight_count, axis=0)
    row_tots = np.sum(straight_count, axis=1)
    PMI = get_pmi(freq)
    return freq, PMI, words, zs, row_tots, col_tots

def make_ngram_mats(D, vocab, K = 3, POS = 'VERB', window_size=0):
    poslst = POS.split(',')
    zs = vocab
    words = sorted(list(set([r['lemma'] for r in D if r['pos'] in poslst])))

    N = len(D)
    freq = np.zeros((len(words), len(zs)))
    straight_count = np.zeros((len(words), len(zs)))
    for i, d in enumerate(D):
        w = d['lemma']
        if d['pos'] in poslst:
            lower = max(0, i-window_size)
            upper = min(i+window_size, N)+1
            for j in range(lower, upper):
                for k in range(K+1):
                    if j+k < len(D):
                        acts = []
                        for idx in range(j, j+k+1):
                            z = get_z(D[idx], MODE)
                            acts.append(z)
                        s = ' '.join(acts)
                        if s in zs:
                            freq[words.index(w)][zs.index(s)] += 1
                            if (i == j):
                                straight_count[words.index(w)][zs.index(s)] += 1
    
    col_tots = np.sum(straight_count, axis=0)
    row_tots = np.sum(straight_count, axis=1)
    
    PMI = get_pmi(freq)
    return freq, PMI, words, zs, row_tots, col_tots

def get_z(D, mode):
    z = D['z']
    if mode == 'endpoints':
        z += "-" if D["end_obj"] == "None" else "+"
    elif mode == 'objects':
        z += D["end_obj"]
    return z

def load_all_spatial_data(data_dir, fnames):
    data = {}
    #for f in os.listdir(data_dir):
    #    frame_file = '%s/%s'%(data_dir, f)
    #    sys.stderr.write("%s\n"%frame_file)
    with open(data_dir) as f:
        for i, line in enumerate(f):
            p, t, d = line.strip().split('\t')
            if i % 500000 == 0:
                sys.stderr.write("%s %s %s\n"%(i, p, t))
            t = t[-1] # want just the task number
            d = json.loads(d)
            if (p, t) not in data:
                data[(p, t)] = {}
            obj = d["name"]
            if (fnames is None) or (obj in fnames):
                if (d["step"] not in data[(p,t)]):
                    data[(p,t)][d["step"]] = {}
                data[(p,t)][d["step"]][obj] = d
    return data

def make_raw_spatial_mats(D, spatial_data, window_size=1):
    all_objs = [ 'Apple', 'Ball', 'Banana', 'Bear', 'Book', 'Bowl', 'Cup', 'Dinosaur', 'Doll', 'Fork', 'Head', 'Knife', 'Lamp', 'LeftHand', 'Plant', 'RightHand', 'Spoon', 'Toy', ]
    N = len(D)
    vocab = ["lx", "ly", "lz", "rx", "ry", "rz"]
    freq = np.zeros((N, len(vocab)))
    lbls = []
    meta = []
    for i, d in enumerate(D):
        start = int(d['step'])
        end = int(D[i+1]['step']) if i < N-1 else N-1
        w = d['lemma']+'_'+d['pos']
        p = d['participant']
        t = d["task"]
        lbls.append(w)
        meta.append(p + ' ' + t + ' ' + '%s-%s'%(start, end))
        fv = []
        for obj in all_objs: #["LeftHand", "RightHand"]:
            #sys.stderr.write('%s %s %s %s %s\n'%(p, t, obj, start, end))
            pdata = spatial_data[(p, t)][obj]
            if (start not in pdata):
                #sys.stderr.write('Skipping start: %s %s %s %s %s\n'%(p, t, obj, start, end))
                fv += [0, 0, 0]
            elif (end not in pdata):
                #sys.stderr.write('Skipping end: %s %s %s %s %s\n'%(p, t, obj, start, end))
                fv += [0, 0, 0]
            else:
                start_frame = pdata[start]
                end_frame = pdata[end]
                for feat in ["relPosX", "relPosY", "relPosZ"]:
                    fv.append(end_frame[feat] - start_frame[feat])
        freq[i, :] = fv
    return freq, lbls, meta, vocab


def make_token_mats(D, _vocab, K = 1, window_size = 1, use_objects=False, mode='raw'):   
    vocab = [v for v in _vocab if len(v.split()) <= K]
    if use_objects:
        objs = list(set([d['end_obj'] for d in D]))
        vocab += objs
    vocabset = set(vocab)
    N = len(D)
    freq = np.zeros((N, len(vocab)))
    lbls = []
    meta = []
    for i, d in enumerate(D):
        start = d['step']
        end = D[i+1]['step'] if i < N-1 else ''
        w = d['lemma']+'_'+d['pos']
        lbls.append(w)
        meta.append(d['participant'] + ' ' + d['task'] + ' ' + '%s-%s'%(start, end))
        lower = max(0, i-window_size)
        upper = min(i+window_size+1, N)
        for j in range(lower, upper):
            if use_objects:
                o = D[j]['end_obj']
                freq[i][vocab.index(o)] += 1
            for k in range(K+1):
                if j+k < len(D):
                    acts = []
                    for idx in range(j, j+k+1):
                        z = get_z(D[idx], mode)
                        acts.append(z)
                    s = ' '.join(acts)
                    if s in vocabset:
                        freq[i][vocab.index(s)] += 1

    #PMI = get_pmi(freq)
    return freq, lbls, meta, vocab

def get_vocabs(data, MODE='raw', K=1, CUTOFF=1):
    # make sequences of states for computing ngrams over
    seqs = {}
    lasts = {}
    for d in data:
        p = d['participant']
        t = d['task']
        k = (p,t)
        if k not in seqs:
            seqs[k] = [[], []]
            lasts[k] = [None, None]
    
        z = get_z(d, MODE)
        if not z == lasts[k][0]:
            seqs[k][0].append(z)
            lasts[k][0] = z
            
        w = '%s_%s'%(d['lemma'], d['pos'])
        if not w == lasts[k][1]:
            seqs[k][1].append(w)
            lasts[k][1] = w
            
    # compute vocabularies to use later
    vocab = [{}, {}] # actions, words
    by_person = [{}, {}] # actions, words
    ns = [K, 1] # how large of ngrams
    
    for k, lsts in seqs.items():
        for i, lst in enumerate(lsts):
            for n in range(ns[i]+1):
                for ngm in ngrams(lst, n):
                    w = ' '.join(['%s'%e for e in ngm])
                    if w not in vocab[i]:
                        vocab[i][w] = 0
                        by_person[i][w] = set()
                    vocab[i][w] += 1
                    by_person[i][w].add(k[0])
                    
    vocab_lsts = [[], []]
    count_lsts = [[], []]
    
    for i, v in enumerate(vocab):
        for w, c in sorted(v.items(), key=lambda e:e[1], reverse=True):
            if len(by_person[i][w]) >= CUTOFF:
                vocab_lsts[i].append(w)
                count_lsts[i].append(c)
    sys.stderr.write("Action Vocab=%d; Word Vocab=%d\n"%(len(vocab_lsts[0]), len(vocab_lsts[1])))
    #print(random.sample(vocab_lsts[0], 10))
    return vocab_lsts, count_lsts

def get_top_words(use_pos):

    #top_words = [w for w in vocab_lsts[1] if w.split('_')[1] == POS][:20] 

    top_words = []
    for l in open('../../nbc/target_words.txt').readlines():
        pos, w = l.strip().split('\t')
        if pos == use_pos:
            top_words.append('%s_%s'%(w, pos))
    return top_words

def filter_mats(X, y, meta, top_words):

    is_top = [yy in top_words for yy in y]
    X_filt = X[is_top, :]
    y_filt = np.array(y)[is_top]
    m_filt = np.array(meta)[is_top]
    return X_filt, y_filt, m_filt

def make_plot(X_train, y_train, m_train, X_dev, y_dev, m_dev, vocab, saveto=None, show=True, supervised=False):
  
    color_scale = px.colors.qualitative.Light24
    color_scale = px.colors.qualitative.Plotly + [e for e in reversed(px.colors.qualitative.Dark24)]

    X_plot = np.vstack((X_train, X_dev))
    y_plot = np.concatenate((y_train, y_dev))
    m_plot = np.concatenate((m_train, m_dev))
    train = np.arange(len(y_train)) # indicies of train instances
    dev = np.arange(len(y_dev)) + len(y_train) # indicies of dev instances
    reducer = TSNE(n_components=2, perplexity=50)
    red = reducer.fit_transform(X_plot)
   
    all_lemmas = [k for k,v in sorted(Counter([y.split('_')[0] for y in y_plot]).items(), key=lambda e:e[1], reverse=True)]

    for split, marker, alpha, edge, size in [(train, 'o', 0.4, 'None', 100), (dev, 'o', 1.0, 'k', 50)]:
        y_split = np.array(y_plot)[split]
        m_split = np.array(m_plot)[split]
        lemmas = [y.split('_')[0] for y in y_split]
        pos = [y.split('_')[1] for y in y_split]
        ps = [y.split(' ')[0] for y in m_split]
        ts = [y.split(' ')[1] for y in m_split]
        pts = [y.rsplit(' ', 1)[0] for y in m_split]
        steps = [y.split(' ')[2] for y in m_split]
    
        ddict = {'x': red[split,0], 'y': red[split,1], 'lemma': lemmas, 'pos': pos,
                 'lemmapos': y_split, 'participant': ps, 'task': ts, 'pt': pts, 'step': steps}
    
        d = pd.DataFrame.from_dict(ddict)
        lemma_colors = [color_scale[all_lemmas.index(l)] for l in lemmas]
        #fig = plt.scatter(d["x"], d["y"])
        fig = plt.scatter(d["x"], d["y"], color=lemma_colors, marker=marker, alpha=alpha, edgecolor=edge, linewidth=1, s=size)

    plt.xlim(np.min(red[:, 0]) - 5, np.max(red[:, 0]) + 35)
    legend_elements = [Patch(facecolor=color_scale[i], label=all_lemmas[i]) for i in range(len(all_lemmas))]
    plt.legend(handles=legend_elements, loc='upper right', ncol=1)
    #plt.legend()

    if saveto is not None:
        plt.savefig("%s.pdf"%saveto, bbox_inches="tight")
    if show:
        plt.show()

def train_test_split(X, y, meta, test_part, dev_part):
    participants = [y.split(' ')[0] for y in meta]
    dev = [p == dev_part for p in participants]
    test = [p == test_part for p in participants]
    train = np.logical_not(np.logical_or(dev, test))
    X_train = X[train, :]
    X_dev = X[dev, :]
    X_test = X[test, :]
    y_train = np.array(y)[train]
    y_dev = np.array(y)[dev]
    y_test = np.array(y)[test]
    m_train = np.array(meta)[train]
    m_dev = np.array(meta)[dev]
    m_test = np.array(meta)[test]
        
    train_tup = X_train, y_train, m_train
    dev_tup = X_dev, y_dev, m_dev
    test_tup = X_test, y_test, m_test

    return train_tup, dev_tup, test_tup

def reduce_mats(train_tup, dev_tup, test_tup, supervised=False, dim=None):
    X_train, y_train, m_train = train_tup
    X_dev, y_dev, m_dev = dev_tup
    X_test, y_test, m_test = test_tup
    sys.stderr.write("Number of classes = %d\nTraining size = %d x %d\n"%(len(set(y_train)), X_train.shape[0], X_train.shape[1]))
    n_obs, n_feats = X_train.shape
    if dim > n_feats:
        sys.stderr.write("Warning, dim > n_feats, setting dim to n_feats (%d)\n"%(n_feats - 1))
        dim = n_feats - 1
    sys.stderr.write("Dim to reduce to = %.03f\n"%dim)

    if supervised:
        r = LinearDiscriminantAnalysis(n_components=dim)
        r.fit(X_train, y_train)
    else:
        r = TruncatedSVD(n_components=dim) # to make as comparable as possible to LDA
        r.fit(X_train)
    mat_train = r.transform(X_train)
    mat_dev = r.transform(X_dev)
    mat_test = r.transform(X_test)
            
    return mat_train, mat_dev, mat_test

def generate_report(mat_train, mat_dev, y_train, y_dev, report_name, N=1):

    nbrs = NearestNeighbors(n_neighbors=N+1).fit(mat_train)

    for nm, mat, lbls in [("train", mat_train, y_train), ("dev", mat_dev, y_dev)]:
        report = open(report_name+"_%s.txt"%nm, 'w')
    
        distances, indices = nbrs.kneighbors(mat)
        by_w = {}
        cms = {}
        for i in range(mat.shape[0]):
            w = lbls[i]
            if w not in by_w:
                by_w[w] = []
                cms[w] = []
            neighbors = [y_train[j] for j in indices[i, 1:]]
            p = sum([1 if u == w else 0 for u in neighbors])/len(neighbors)
            by_w[w].append(p)
            cms[w] += neighbors

        all_words = sorted(cms.keys())
        cm = np.zeros((len(all_words), len(all_words)))
        for i, w in enumerate(all_words):
            N = len(cms[w])
            for j, w2 in enumerate(all_words):
                cm[i][j] = (cms[w].count(w2))/N
        sns.heatmap(cm, cmap="Blues", annot=True, fmt=".01f", cbar=False)
        ticks = [w.split('_')[0] for w in all_words]
        plt.xticks(np.arange(len(all_words))+0.5, ticks, rotation=90)
        plt.yticks(np.arange(len(all_words))+0.5, ticks, rotation=0)
        fig_name = report_name.replace('reports', 'figures')
        plt.savefig(fig_name+"_cm_%s.pdf"%nm)
        plt.clf()
        #plt.show()
        
        macro = []
        micro = []
        tot = 0.
        for w, lst in sorted(by_w.items(), key=lambda e:len(e[1]), reverse=True):
            n = len(lst)
            tp = sum(lst)
            p = tp / n
            macro.append(p)
            micro.append(tp)
            tot += n
            report.write('%s\t%.02f\t%s\n'%(w, p, n))
        
        macro_avg = sum(macro)/len(macro)
        micro_avg = sum(micro)/tot
        report.write("Macro: %.02f\n"%(macro_avg))
        report.write("Micro: %.02f\n"%(micro_avg))
        cols = report_name.split('/')[-1].split('_') 
        cols = [e.split('=')[1] for e in cols]
        sys.stderr.write("%.02f\t%.02f\t%s\t%s\n"%(macro_avg, micro_avg, nm, '\t'.join(cols)))
        sys.stderr.write("%.02f\t%.02f\t%s\t%s"%(macro_avg, micro_avg, nm, '\t'.join(cols)))
        report.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate word clusters for GAILA Milestone 4 report.')
    parser.add_argument('--aligned_data', type=str, default='aigned_data.tsv',
    		    help='file containing aligned data')
    parser.add_argument('--mode', type=str, default='raw',
		    help='how to represent states, one of "raw", "endpoints", "objects", "random"')
    parser.add_argument('--pos', type=str, default="VERB",
		    help='which POS to use')
    parser.add_argument('--window', type=int, default=0,
		    help='word use +/- how many states?')
    parser.add_argument('--k', type=int, default=1,
		    help='how large of ngrams to use')
    parser.add_argument('--cutoff', type=int, default=0,
		    help='how many unique people need do actions have to occur with?')
    parser.add_argument('--objs', type=str, default="false",
		    help='whether to include "used objects" as features (separate from the states)')
    parser.add_argument('--supervised', type=str, default="false",
		    help='whether the dimensionality reduction has access to labels')
    parser.add_argument('--reduction', type=str, default="false",
		    help='whether to run dimensionality reduction')
    parser.add_argument('--dim', type=int, default=-1,
		    help='embedding size for dimensionality reduction or -1 if no reduction')
    parser.add_argument('--plot', action="store_true", default=False,
		    help='plot data')
    parser.add_argument('--dev_participant', type=str, default='4_2b',
		    help='hold out participant for dev')
    parser.add_argument('--test_participant', type=str, default='1_1a',
		    help='hold out participant for test')
    parser.add_argument('--rep', type=int, default=0,
    		    help='use to log multiple identical runs')

    args = parser.parse_args()

    #np.random.seed(args.seed)

    OBJS = args.objs == "true"
    SUP = args.supervised == "true"
    RED = args.reduction == "true"

    _data = [row for row in csv.DictReader(open(args.aligned_data), delimiter='\t')]

    vocab_lsts = get_vocabs(_data, MODE=args.mode, K=args.k, CUTOFF=args.cutoff)
    X, y, meta, vocab = make_token_mats(_data, vocab_lsts[0], K = args.k,
                                    window_size = args.window, use_objects=OBJS,
				    mode=args.mode)

    if args.mode == 'random':
        rows, cols = X.shape
        X = np.random.rand(rows, cols)
    
    if args.mode == 'oracle':
        rows, cols = X.shape
        words = sorted(list(set(y)))
        X = np.random.rand(rows, len(words))
        #X = np.zeros((rows, len(words)))
        for i, w in enumerate(y):
            X[i, words.index(w)] = 10

    top_words = get_top_words(args.pos)
    #top_words = sorted(list(set(y)))
    X_filt, y_filt, m_filt = filter_mats(X, y, meta, top_words)

    inp = args.aligned_data.split('.')[0].split('/')[-1]
    repdir = 'reports/%s'%inp
    figdir = 'figures/%s'%inp
    if not os.path.exists(repdir):
        os.makedirs(repdir)
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    name = '%s/pos=%s_mode=%s_obj=%s_win=%s_k=%s_reduce=%s_supervise=%s_dim=%s_rep=%d'%(repdir, args.pos,
		    args.mode, OBJS, args.window, args.k, RED, SUP, args.dim, args.rep)
    figname = '%s/pos=%s_mode=%s_obj=%s_win=%s_k=%s_reduce=%s_supervise=%s_dim=%s_rep=%d'%(figdir, args.pos,
		    args.mode, OBJS, args.window, args.k, RED, SUP, args.dim, args.rep)

    X_train, X_dev, X_test, y_train, y_dev, y_test, m_train, m_dev, m_test = fit_model(
		    X_filt, y_filt, m_filt, args.test_participant, args.dev_participant, dim=args.dim, dr=RED, supervised=SUP)

    generate_report(X_train, X_dev, y_train, y_dev, name)
   
    if args.plot:
        print("Generating plots...", end='')
        make_plot(X_train, y_train, m_train, X_dev, y_dev, m_dev,
			vocab, saveto = figname, show = True, supervised = SUP)
        print("done.")

    


            
