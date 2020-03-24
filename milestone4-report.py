"""
Script for generating nearest neighbors/cluster analyses for GAILA Milestone 4 report.
"""

import sys
import csv
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

def make_token_mats(D, _vocab, K = 1, window_size = 1, use_objects=True, mode='raw'):   
    vocab = [v for v in _vocab if len(v.split()) <= K+1]
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
        upper = min(i+window_size, N)
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
    
    for i, v in enumerate(vocab):
        for w, c in sorted(v.items(), key=lambda e:e[1], reverse=True):
            if len(by_person[i][w]) >= CUTOFF:
                vocab_lsts[i].append(w)
    return vocab_lsts

def get_top_words(use_pos):

    #top_words = [w for w in vocab_lsts[1] if w.split('_')[1] == POS][:20] 

    top_words = []
    for l in open('../nbc/target_words.txt').readlines():
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

def make_plot(X_plot, y_plot, m_plot, vocab, saveto=None, show=True):

    reducer = TSNE(n_components=2)
    red = reducer.fit_transform(X_plot)
    
    lemmas = [y.split('_')[0] for y in y_plot]
    pos = [y.split('_')[1] for y in y_plot]
    ps = [y.split(' ')[0] for y in m_plot]
    ts = [y.split(' ')[1] for y in m_plot]
    pts = [y.rsplit(' ', 1)[0] for y in m_plot]
    steps = [y.split(' ')[2] for y in m_plot]
    
    ddict = {'x': red[:,0], 'y': red[:,1], 'lemma': lemmas, 'pos': pos,
             'lemmapos': y_plot, 'participant': ps, 'task': ts, 'pt': pts, 'step': steps}
    
    top_feat = [vocab[i] for i in np.argmax(X_plot, axis=1)]
    ddict['top'] = top_feat
        
    feats = ['participant', 'top', 'step', 'pt'] 
    
    if red.shape[1] == 3:
        ddict['z'] = red[:,2]
        d = pd.DataFrame.from_dict(ddict)
        fig = px.scatter_3d(d, x="x", y="y", z="z", color='lemma',
                     hover_data=feats)
    else:
        d = pd.DataFrame.from_dict(ddict)
        fig = px.scatter(d, x="x", y="y", color='lemma', hover_data=feats,
                        color_discrete_sequence=px.colors.qualitative.Light24)
       
    if saveto is not None:
        fig.write_image("%s.pdf"%saveto)
    if show:
        fig.show()

def generate_report(X, y, meta, report_name, reduction=None, supervised=False, N=5):

    participants = [y.split(' ')[0] for y in meta]
    dev = [p == '4_2b' for p in participants]
    test = [p == '6_2c' for p in participants]
    train = np.logical_not(np.logical_or(dev, test))
    X_train = X[train, :]
    X_dev = X[dev, :]
    X_test = X[test, :]
    y_train = np.array(y)[train]
    y_dev = np.array(y)[dev]
    y_test = np.array(y)[test]

    if supervised:
        if reduction is not None:
            r = LinearDiscriminantAnalysis(n_components=reduction)
        else:
            r = LinearDiscriminantAnalysis()
        r.fit(X_train, y_train)
        mat_train = r.transform(X_train)
        mat_dev = r.transform(X_dev)
    else:
        if reduction is not None:
            r = TruncatedSVD(n_components=reduction) # to make as comparable as possible to LDA
            r.fit(X_train)
            mat_train = r.transform(X_train)
            mat_dev = r.transform(X_dev)
        else:
            mat_train = X_train
            mat_dev = X_dev
    
    nbrs = NearestNeighbors(n_neighbors=N+1).fit(mat_train)
  

    for nm, mat, lbls in [("train", mat_train, y_train), ("dev", mat_dev, y_dev)]:
        report = open(report_name+"_%s.txt"%nm, 'w')
    
        distances, indices = nbrs.kneighbors(mat)
        by_w = {}
        for i in range(mat.shape[0]):
            w = lbls[i]
            if w not in by_w:
                by_w[w] = []
            neighbors = [y_train[j] for j in indices[i, 1:]]
            p = sum([1 if u == w else 0 for u in neighbors])/len(neighbors)
            by_w[w].append(p)
        
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
        cols = report_name.split('/')[1].split('_') 
        cols = [e.split('=')[1] for e in cols]
        sys.stderr.write("%.02f\t%.02f\t%s\t%s\n"%(macro_avg, micro_avg, nm, '\t'.join(cols)))
        print("%.02f\t%.02f\t%s\t%s"%(macro_avg, micro_avg, nm, '\t'.join(cols)))
        report.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate word clusters for GAILA Milestone 4 report.')
    parser.add_argument('--mode', type=str, default='raw', help='how to represent states, one of "raw", "endpoints", "objects", "random"')
    parser.add_argument('--pos', type=str, default="VERB", help='which POS to use')
    parser.add_argument('--window', type=int, default=1, help='word use +/- how many states?')
    parser.add_argument('--k', type=int, default=2, help='how large of ngrams to use')
    parser.add_argument('--cutoff', type=int, default=2, help='how many unique people need do actions have to occur with?')
    parser.add_argument('--objs', type=str, default="false", help='whether to include "used objects" as features (separate from the states)')
    parser.add_argument('--supervised', type=str, default="false", help='whether the dimensionality reduction has access to labels')
    parser.add_argument('--reduction', type=int, default=-1, help='embedding size for dimensionality reduction or -1 if no reduction')

    args = parser.parse_args()

    OBJS = args.objs == "true"
    SUP = args.supervised == "true"
    RED = None if args.reduction < 0 else args.reduction

    #print("Reading data...", end='')
    _data = [row for row in csv.DictReader(open('aligned_data.tsv'), delimiter='\t')]
    #print("done.")

    #print("Generating matrices...", end='')
    vocab_lsts = get_vocabs(_data, MODE=args.mode, K=args.k, CUTOFF=args.cutoff)
    X, y, meta, vocab = make_token_mats(_data, vocab_lsts[0], K = args.k,
                                    window_size = args.window, use_objects=OBJS, mode=args.mode)

    if args.mode == 'random':
        rows, cols = X.shape
        X = np.random.rand(rows, cols)

    top_words = get_top_words(args.pos)
    X_filt, y_filt, m_filt = filter_mats(X, y, meta, top_words)
    #print("done.")

    name = 'pos=%s_mode=%s_obj=%s_win=%s_k=%s_reduce=%s_supervise=%s'%(args.pos, args.mode, OBJS, args.window, args.k, RED, SUP)

    #print("Running eval...", end='')
    generate_report(X_filt, y_filt, m_filt, 'reports/%s'%name, reduction=RED, supervised=SUP)
    #print("done.")
    
    #print("Generating plots...", end='')
    #make_plot(X_filt, y_filt, m_filt, vocab, saveto = 'figures/%s'%name, show = True, supervised = SUP)
    #print("done.")

    


            
