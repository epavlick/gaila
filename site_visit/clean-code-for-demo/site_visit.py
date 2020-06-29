"""
Script for generating nearest neighbors/cluster analyses using just raw spatial features
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
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import collections
from feature_functions import *

def load_all_spatial_data(data_dir, fnames):
    data = {}
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

def _make_vocab_sequences(data):
    # make sequences of states for computing ngrams over
    seqs = {}
    lasts = {}
    for d in data:
        p = d['participant']
        t = d['task']
        k = (p,t)
        if k not in seqs:
            seqs[k] = []
            lasts[k] = None
    
        w = '%s_%s'%(d['lemma'], d['pos'])
        if not w == lasts[k]:
            seqs[k].append(w)
            lasts[k] = w
    return seqs

def get_vocabs(data, MODE='raw', K=1, CUTOFF=1):
    """
    This function returns the list of words in the vocab and the number of times each was used.
    Its a bit hacky because its based on the old files that included the BPARHMM state symbols,
    but it doesn't actually use those symbols. Should rewrite all the code to take in just the
    raw spatial files.
    """
    seqs = _make_vocab_sequences(data)
    vocab = {} 
    by_person = {} 
    
    for k, lsts in seqs.items():
        for i, lst in enumerate(lsts):
            for n in range(K+1):
                for ngm in ngrams(lst, n):
                    w = ' '.join(['%s'%e for e in ngm])
                    if w not in vocab:
                        vocab[w] = 0
                        by_person[w] = set()
                    vocab[w] += 1
                    by_person[w].add(k[0])
                    
    vocab_lst = []
    count_lst = []
    
    for w, c in sorted(vocab.items(), key=lambda e:e[1], reverse=True):
        if len(by_person[w]) >= CUTOFF:
            vocab_lst.append(w)
            count_lst.append(c)

    sys.stderr.write("Word Vocab=%d\n"%(len(vocab_lst)))
    return vocab_lst, count_lst

def get_top_words(use_pos):

    #top_words = [w for w in vocab_lsts[1] if w.split('_')[1] == POS][:20] 

    top_words = []
    for l in open('resources/target_words.txt').readlines():
        pos, w = l.strip().split('\t')
        if pos == use_pos:
            top_words.append('%s_%s'%(w, pos))
    return top_words

def filter_mats(data_m, top_words):

    is_top = [yy in top_words for yy in data_m.y]
    X_filt = data_m.X[is_top, :]
    y_filt = np.array(data_m.y)[is_top]
    m_filt = np.array(data_m.meta)[is_top]
    return DataMat(X_filt, y_filt, m_filt)

def train_test_split(D, test_part, dev_part):
    participants = [y.split(' ')[0] for y in D.meta]
    dev = [p in dev_part for p in participants]
    test = [p in test_part for p in participants]
    train = np.logical_not(np.logical_or(dev, test))
    X_train = D.X[train, :]
    X_dev = D.X[dev, :]
    X_test = D.X[test, :]
    y_train = np.array(D.y)[train]
    y_dev = np.array(D.y)[dev]
    y_test = np.array(D.y)[test]
    m_train = np.array(D.meta)[train]
    m_dev = np.array(D.meta)[dev]
    m_test = np.array(D.meta)[test]
        
    train = DataMat(X_train, y_train, m_train)
    dev = DataMat(X_dev, y_dev, m_dev)
    test = DataMat(X_test, y_test, m_test)

    return train, dev, test

def reduce_mats(train, dev, test, supervised=False, dim=None):
    sys.stderr.write("Number of classes = %d\nTraining size = %d x %d\n"%(len(set(train.y)), train.X.shape[0], train.X.shape[1]))
    n_obs, n_feats = train.X.shape
    if dim > n_feats:
        sys.stderr.write("Warning, dim > n_feats, setting dim to n_feats (%d)\n"%(n_feats - 1))
        dim = n_feats - 1
    sys.stderr.write("Dim to reduce to = %.03f\n"%dim)

    if supervised:
        r = LinearDiscriminantAnalysis(n_components=dim)
        r.fit(train.X, train.y)
    else:
        r = TruncatedSVD(n_components=dim) 
        r.fit(train.X)
            
    return r 


def eval_on_random(config, spatial_data, vocab_lsts, count_lsts, use_cnn=False):
    
    #_data = [row for row in csv.DictReader(open(config['DATA']), delimiter='\t')]
    sample = []
    p = config['DEV'][0]
    for t in range(1,7):
        for step in spatial_data[(p,str(t))]:
            if step % 450 == 0:
                sample.append({'participant': p, 'task': str(t),
                              'step': step, 'lemma': 'NA', 'pos': 'NA'})

    sys.stderr.write("Data: %d items\n"%len(sample))
    
    if use_cnn:
        dat = _make_pretrained_cnn_mats(sample,
                                        window_size = config['WINDOW'], logdir=None)

    else:
        dat, _ = make_raw_spatial_mats(sample, spatial_data,
                                    feat_lst=config['FEATURES'],
                                    basic_feats=config['BASIC_FEATS'],
                                    use_objs=config['OBJS'],
                                    use_most_moving=config['MOST_MOVING'],
                                    window_size = config['WINDOW'],
                                    logdir=None)
                                    
    return dat

def prep_data(config, spatial, logdir, use_cnn = False):

    _data = [row for row in csv.DictReader(open(config['DATA']), delimiter='\t')]
    sys.stderr.write("Data: %d items\n"%len(_data))
    
    vocab_lsts, count_lsts = get_vocabs(_data, MODE=config['MODE'],
                                        K=config['K'], CUTOFF=config['CUTOFF'])
                                        
    top_words = get_top_words(config['POS'])
    _data = [d for d in _data if '%s_%s'%(d['lemma'], d['pos']) in top_words]
    sys.stderr.write("Filtered Data: %d items\n"%len(_data))
    
    
    if use_cnn:
        featurized_data = _make_pretrained_cnn_mats(_data, window_size = config['WINDOW'],
                                                    logdir=logdir)
        fnames = []
    else:
        featurized_data, dv = make_raw_spatial_mats(_data, spatial,
                                                             feat_lst=config['FEATURES'],
                                                             basic_feats=config['BASIC_FEATS'],
                                                             use_objs=config['OBJS'],
                                                             use_most_moving=config['MOST_MOVING'],
                                                             window_size = config['WINDOW'],
                                                             logdir=logdir)
        fnames = dv.get_feature_names()

    if config['MODE'] == 'random':
        rows, cols = featurized_data.X.shape
        featurized_data.X = np.random.rand(rows, cols)
    if config['MODE'] == 'oracle':
        rows, cols = featurized_data.X.shape
        words = sorted(list(set(featurized_data.y)))
        featurized_data.X = np.random.rand(rows, len(words))
        for i, w in enumerate(featurized_data.y):
            featurized_data.X[i, words.index(w)] = 1

    return vocab_lsts, count_lsts, top_words, featurized_data, fnames


def filter_and_dedup(data_mat, top_words):
    data_mat_filtered = filter_mats(data_mat, top_words)
    
    dedup = []
    seen = set()
    for i, (mm, yy) in enumerate(zip(data_mat_filtered.meta, data_mat_filtered.y)):
        part, task, step, steps = mm.split()
        if (part, task, yy) not in seen:
            seen.add((part, task, yy))
            dedup.append(i)

    data_mat_filtered.X = data_mat_filtered.X[dedup]
    data_mat_filtered.y = data_mat_filtered.y[dedup]
    data_mat_filtered.meta = data_mat_filtered.meta[dedup]
    
    return data_mat_filtered

def name_from_config(config):
    fnames = ['k', 'win']
    fs = ['K', 'WINDOW']
    
    return '_'.join(['%s=%s'%(fnames[i],
                              config[fs[i]]) for i in range(len(fs))])

def make_directories(config, rootdir):
    
    make = config['LOG']
    
    figdir = rootdir + 'red=%s_dim=%s_sup=%s/figures'%(config['RED'],
                                                       config['DIM'], config['SUP'])
    repdir = rootdir + 'red=%s_dim=%s_sup=%s/reports'%(config['RED'],
                                                   config['DIM'], config['SUP'])
    if make:
        if not os.path.exists(figdir):
            os.makedirs(figdir)
        if not os.path.exists(repdir):
            os.makedirs(repdir)

    return repdir, figdir

def avg_mat(data_mat, words):
    """
    Returns the averaged/centroid vector for each word. We used this
    to compute a simple K-means classifier for predicting the word.
    """
    use_X = data_mat.X
    use_y = data_mat.y
    use_m = data_mat.meta
    
    n_obs, n_feat = use_X.shape
    mat = np.zeros((len(words), n_feat))
    std = np.zeros((len(words), n_feat))
    for i, w in enumerate(words):
        avg = use_X[use_y == w, :].mean(axis=0)
        mat[i, :] = avg
        std[i, :] = use_X[use_y == w, :].std(axis=0)
    return mat, std

def _simulate_random(dv_lbl, tr_lbl):
    rand = []
    rand_p3 = []
    for _ in range(100):
        rr = []
        pp = []
        for _ in range(len(dv_lbl)):
            rank = random.randint(0, len(tr_lbl))
            rr.append(1./(rank+1))
            pp.append(1 if rank < 3 else 0)
        rand.append(sum(rr)/len(rr))
        rand_p3.append(sum(pp)/len(pp))
    return rand, rand_p3

def generate_results_reports(tr, dv, tr_lbl, dv_lbl, tr_meta, dv_meta, outdir, save=False):
    """"
    Generates all of the results using our hacky automatic eval, which measures how well the model predicts
    the exact word that the human said at a given frame. We use MRR/P@K metrics for this.
    """
    sims = cosine_similarity(dv, tr)
    rand = []
    debug_lines = []
    N = len(dv_lbl)
    by_lbl = {}
    by_w = {w: [] for w in set(dv_lbl)}
    pat3 = {w: [] for w in set(dv_lbl)}
    
    cm = np.zeros((len(tr_lbl), len(tr_lbl)))
    cm2 = np.zeros((len(tr_lbl), len(tr_lbl)))

    for i, w in enumerate(dv_lbl):
   
        # most similar word
        max_idxs = [e for e in reversed(sims[i, :].argsort())] # to get descending order
        max_idx = max_idxs[0]
        
        # find the rank of the most-similar instance of the true word
	# this is a little awkward because its written so that it can accommodate the case 
	# in which there might be more than one vector for a given word in train (i.e. for 
	# a "proper" nearest neighbors classifier)
        matches = [idx for idx, u in enumerate(tr_lbl) if u == w]
        ranked_correct = [(idx, max_idxs.index(idx)) for idx in matches]
        top_correct_idx, top_correct_rank = sorted(ranked_correct, key=lambda e:e[1])[0]
        
        by_w[w].append(1./(1+top_correct_rank))
        pat3[w].append(1 if top_correct_rank < 3 else 0)
        
        cells = {}
        cells['word'] = w
        p, t, step, steps = dv_meta[i].split()
        cells['participant'] = p
        cells['task'] = t
        cells['steps'] = steps
        cells['correct'] = (tr_lbl[max_idx] == w)
        cells['pred'] = (tr_lbl[max_idx])
        cells['rank_w'] = top_correct_rank
        cells['sim_w'] = '%.06f'%sims[i, top_correct_idx]
        cells['sim_pred'] = '%.06f'%sims[i, max_idx]
        cm[tr_lbl.index(w), max_idx] += 1
        for j, s in enumerate(sims[i, :]):
            cm2[tr_lbl.index(w), j] += s
        debug_lines.append((w, cells))

    rand, rand_p3 = _simulate_random(dv_lbl, tr_lbl)
    rand = sorted(rand)

    out = open('%s/summary.txt'%outdir, 'w')
    out.write("Worst-Case Baseline:\t%.02f\n"%(1/len(tr_lbl)))
    out.write("Random Baseline:\t%.02f (%.02f -- %.02f)\t%.02f\n"%(sum(rand)/len(rand),
                                rand[0], rand[-1], sum(rand_p3)/len(rand_p3)))
    print("Worst-Case Baseline:\t%.02f"%(1/len(tr_lbl)))
    print("Random Baseline:\t%.02f (%.02f -- %.02f)\t%.02f"%(sum(rand)/len(rand), rand[0], rand[-1], sum(rand_p3)/len(rand_p3)))

    lines = []
    macro = []
    p3s = []
    for w, lst in sorted(by_w.items()):
        mrr = sum(lst)/len(lst)
        p3 = sum(pat3[w])/len(pat3[w])
        macro.append(mrr)
        p3s.append(p3)
        lines.append('%s\t%.02f\t%.02f\t%d'%(w, mrr, p3, len(lst)))
    out.write("Macro Avg. MRR:\t%.02f\t%.02f\n"%(sum(macro)/len(macro), sum(p3s)/len(p3s)))
    print("Macro Avg. MRR:\t%.02f\t%.02f"%(sum(macro)/len(macro), sum(p3s)/len(p3s)))
    
    for line in lines:
        out.write(line+'\n')
    out.close()

    writer = csv.DictWriter(open("%s/detail.tsv"%outdir, 'w'),
                        fieldnames=['word','correct','pred','rank_w','sim_w','sim_pred',
                                    'participant', 'task', 'steps'],
                        delimiter='\t')
    writer.writeheader()
    for w, line in sorted(debug_lines, key=lambda e:e[0]):
        writer.writerow(line)

    plt.figure(figsize=(11,4))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, cmap="Blues")
    plt.xticks(np.arange(len(tr_lbl))+0.5, tr_lbl, rotation='vertical')
    plt.yticks(np.arange(len(tr_lbl))+0.5, tr_lbl, rotation='horizontal')
    plt.title("Confusion Matrix: Top-Ranked")
    plt.subplot(1, 2, 2)
    sns.heatmap(cm2, cmap="Blues")
    plt.xticks(np.arange(len(tr_lbl))+0.5, tr_lbl, rotation='vertical')
    plt.yticks(np.arange(len(tr_lbl))+0.5, tr_lbl, rotation='horizontal')
    plt.title("Confusion Matrix: Similarity Weighted")
    plt.savefig(outdir+"/confusion.pdf", bbox_inches="tight")
    plt.show()


