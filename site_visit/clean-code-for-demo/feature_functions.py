import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
import json
from sklearn.feature_extraction import DictVectorizer


class DataMat:
    def __init__(self, _X, _y, _meta):
        self.X = _X
        self.y = _y
        self.meta = _meta

def _max_dist_change_feats(ref, mat):
    dists = distance_matrix(ref.reshape(1, -1), mat)
    
    max_idx = [e for e in reversed(dists[0].argsort())][0]
    mx = dists[0, max_idx]
    
    mxx, mxy, mxz = mat[max_idx, :]
    names = ["dist", "x", "y", "z"]
    feats = [mx, mxx, mxy, mxz]
    return feats, names

def _dist_change_feats(s, e):
    
    d = euclidean(s, e)
    
    names = ["dist", "x", "y", "z"]
    feats = [d] + [s[i] - e[i] for i in range(3)]
    return feats, names

def _trajectory_feats(span, dims):
    
    feats = []
    names = []
    
    # start value of each dim
    names += ["start_%s"%d for d in dims]
    feats += list(span[0, :])
    
    # peak value of each dim
    peak = span.max(axis=0)
    names += ["peak_%s"%d for d in dims]
    feats += list(peak)
    
    # trough value of each dim
    trough = span.min(axis=0)
    names += ["trough_%s"%d for d in dims]
    feats += list(trough)
    
    # end value of each dim
    names += ["end_%s"%d for d in dims]
    feats += list(span[-1, :])
    
    # break trajectory into 3 pieces, start->key point 1, kp1 -> kp2, kp2->end
    peak_idx = np.argmax(span, axis=0)
    trough_idx = np.argmin(span, axis=0)
    for i, d in enumerate(dims):
        start = span[0, i]
        end = span[-1, i]
        if peak_idx[i] < trough_idx[i]:
            kp1 = peak[i]
            kp2 = trough[i]
        else:
            kp2 = peak[i]
            kp1 = trough[i]
        names += ['st-kp1_%s'%d, 'kp1-kp2_%s'%d, 'kp2-end_%s'%d]
        feats += [kp1-start, kp2-kp1, end-kp2]
        
        names += ['st-end_%s'%d]
        feats += [end-start]
    
    return feats, names

def _avg_feats(span, dims):
    
    avg = span.mean(axis=0)
    mn = span.min(axis=0)
    mx = span.max(axis=0)
    
    names = ["%s_avg"%d for d in dims]
    names += ["%s_min"%d for d in dims]
    names += ["%s_max"%d for d in dims]
    names += ["%s_start"%d for d in dims]
    names += ["%s_end"%d for d in dims]
    
    feats = [avg[i] for i in range(len(dims))]
    feats = [mn[i] for i in range(len(dims))]
    feats = [mx[i] for i in range(len(dims))]
    
    feats += list(span[0, :])
    feats += list(span[-1, :])
    return feats, names

def _var_feats(span, dims):
    
    avg = span.var(axis=0)
    
    names = ["%s_var"%d for d in dims]
    feats = [avg[i] for i in range(len(dims))]
    return feats, names

def _dist_to_obj_feats(obj, ref):
    
    at_start = euclidean(obj[0], ref[0])
    at_end = euclidean(obj[-1], ref[-1])
    
    N, _ = obj.shape
    
    diff = obj - ref
    sq = (diff * diff).sum(axis=1) # squared differences
    dists = np.sqrt(sq)
    
    avg = dists.mean()
    var = dists.var()
    mn = dists.min()
    mx = dists.max()
    argmn = np.argmin(dists, axis=0)
    argmx = np.argmax(dists, axis=0)
    #print("mn=%.02f argmn=%.02f mx=%.02f argmx=%.02f"%(mn, argmn, mx, argmx))
    
    names = ["start", "end", "mean", 'var', "min", "max", "min_idx", "max_idx"]
    feats = [at_start, at_end, avg, var, mn, mx, argmn/N, argmx/N]
    return feats, names

def get_features(span, all_feats, all_obj_spans, use=None):
    
    feats = []
    names = []
    
    # distance to the floor
    if (use is None) or ('dist_to_surfaces' in use):
        px = all_feats.index('posX')
        py = all_feats.index('posY')
        pz = all_feats.index('posZ')
        this = span[:, [px, py, pz]]
        
        ref = all_obj_spans['Floor'][:, [px, py, pz]]
        fs, ns = _dist_to_obj_feats(this, ref)
        names += ["dist_to_floor_"+e for e in ns]
        feats += fs
        
        ref = all_obj_spans['Counter'][:, [px, py, pz]]
        fs, ns = _dist_to_obj_feats(this, ref)
        names += ["dist_to_counter_"+e for e in ns]
        feats += fs
    
    # distance to the right hand
    if (use is None) or ('dist_to_rhand' in use):
        px = all_feats.index('posX')
        py = all_feats.index('posY')
        pz = all_feats.index('posZ')
        this = span[:, [px, py, pz]]
        ref = all_obj_spans['RightHand'][:, [px, py, pz]]
        fs, ns = _dist_to_obj_feats(this, ref)
        names += ["dist_to_rhand_"+e for e in ns]
        feats += fs

    # distance to the left hand
    if (use is None) or ('dist_to_lhand' in use):
        px = all_feats.index('posX')
        py = all_feats.index('posY')
        pz = all_feats.index('posZ')
        this = span[:, [px, py, pz]]
        ref = all_obj_spans['LeftHand'][:, [px, py, pz]]
        fs, ns = _dist_to_obj_feats(this, ref)
        names += ["dist_to_lhand_"+e for e in ns]
        feats += fs

    # distance to head
    if (use is None) or ('dist_to_head' in use):
        px = all_feats.index('posX')
        py = all_feats.index('posY')
        pz = all_feats.index('posZ')
        this = span[:, [px, py, pz]]
        ref = all_obj_spans['Head'][:, [px, py, pz]]
        fs, ns = _dist_to_obj_feats(this, ref)
        names += ["dist_to_head_"+e for e in ns]
        feats += fs

    # average velocity
    if (use is None) or ('vel_avg' in use):
        px = all_feats.index('velX')
        py = all_feats.index('velY')
        pz = all_feats.index('velZ')
        fs, ns = _avg_feats(span[:, [px, py, pz]], ['x', 'y', 'z'])
        if np.isnan(fs).any():
            print("NaNs!! avg vel %s"%(' '.join(ns)), fs)
        names += ["vel_avg_"+e for e in ns]
        feats += fs

    # variation in velocity
    if (use is None) or ('vel_var' in use):
        px = all_feats.index('velX')
        py = all_feats.index('velY')
        pz = all_feats.index('velZ')
        fs, ns = _var_feats(span[:, [px, py, pz]], ['x', 'y', 'z'])
        if np.isnan(fs).any():
            print("NaNs!! avg vel %s"%(' '.join(ns)), fs)
        names += ["vel_var_"+e for e in ns]
        feats += fs

    # trajectory data start, peak, trough, end
    if (use is None) or ('trajectory' in use):
        px = all_feats.index('posX')
        py = all_feats.index('posY')
        pz = all_feats.index('posZ')
        fs, ns = _trajectory_feats(span[:, [px, py, pz]], ['x', 'y', 'z'])
        if np.isnan(fs).any():
            print("NaNs!! %s"%(' '.join(ns)), fs)
        names += ["traj_"+e for e in ns]
        feats += fs

    # average relative position
    if (use is None) or ('relPos_avg' in use):
        px = all_feats.index('relPosX')
        py = all_feats.index('relPosY')
        pz = all_feats.index('relPosZ')
        fs, ns = _avg_feats(span[:, [px, py, pz]], ['x', 'y', 'z'])
        if np.isnan(fs).any():
            print("NaNs!! avg vel %s"%(' '.join(ns)), fs)
        names += ["relPos_avg_"+e for e in ns]
        feats += fs

    return feats, names

def get_most_moving(obj_spans, fnames):
    max_objs = []
    x = fnames.index('velX')
    y = fnames.index('velY')
    z = fnames.index('velZ')
    for obj, span in obj_spans.items():
        try:
            vals, _ = _max_dist_change_feats(span[0, [x, y, z]], span[:, [x, y, z]])
        except IndexError:
            print(span)
        d = vals[0] # euclidean distance
        max_objs.append((obj, d))
    rank = sorted(max_objs, key=lambda e:e[1], reverse=True)
    return [o for o, _ in rank if 'Hand' not in o][0]

def make_raw_spatial_mats(D, spatial_data, feat_lst=[], basic_feats=[],
                           use_objs=[], use_most_moving=True, window_size=100,
                           logdir=None):
    
    all_objs = [l.strip() for l in open('resources/obj_names.txt').readlines()]
    all_feats = ["relPosX", "relPosY", "relPosZ", "posX", "posY", "posZ",
                 'velX', 'velY', 'velZ', 'relVelX', 'relVelY', 'relVelZ']


    N = len(D)
    feats = []
    lbls = []
    meta = []
    if logdir is not None:
        logfile = open('%s/feature_dump.tsv'%logdir, 'w')
    for i, d in enumerate(D):
        sidx = int(d['step'])
        start = sidx #-window_size
        end = sidx+window_size
        L = 1+end-start
        w = d['lemma']+'_'+d['pos']
        p = d['participant']
        t = d["task"]
        lbls.append(w)
        meta.append(p + ' ' + t + ' %s '%sidx + '%s-%s'%(start, end))
        fv = {}
        obj_spans = {o: [] for o in all_objs}
        for si, step in enumerate(range(start, end+1)):
            if step in spatial_data[(p, t)]:
                sdata = spatial_data[(p, t)][step]
                for obj in all_objs:
                    if obj in sdata:
                        frame = sdata[obj]
                        row = [frame[e] for e in all_feats]
                        if not(np.isnan(row).any()):
                            obj_spans[obj].append(row)
        obj_spans = {o: np.array(v) for o, v in obj_spans.items() if len(v)>0}
        covered = set()
        for obj in use_objs:
            x, names = get_features(obj_spans[obj], all_feats, obj_spans, use=feat_lst)
            fv.update({obj+'_'+n: xx for (xx, n) in zip(x, names)})
            covered.add(obj)
        if use_most_moving:
            most_moving_obj = get_most_moving(obj_spans, all_feats)
            x, names = get_features(obj_spans[most_moving_obj], all_feats, obj_spans, use=feat_lst)
            fv.update({'most_moving_'+n: xx for (xx, n) in zip(x, names)})
            covered.add(most_moving_obj)

        for obj in all_objs:
            #if obj not in obj_spans:
            #    print("no object data for %s"%obj)
            if (obj not in covered) and (obj in obj_spans):
                x, names = get_features(obj_spans[obj], all_feats,
                                obj_spans, use=basic_feats)
                fv.update({obj+'_'+n: xx for (xx, n) in zip(x, names)})

        feats.append(fv)
        if logdir is not None:
            logfile.write('%s\t%s\t%s\t%s-%s\t%s\tfdim=%d\t%s\n'%(w, p, t, start, end, most_moving_obj,
                                                              len(fv.keys()), json.dumps(fv)))
    if logdir is not None:
        logfile.close()
    dv = DictVectorizer(sparse=False)
    Xmat = dv.fit_transform(feats)
    print("feats", Xmat.shape)
    return DataMat(Xmat, lbls, meta), dv


def _make_pretrained_cnn_mats(D, window_size, logdir=None):
    
    sys.stderr.write("Loading pretrained embeddings\n")
    EMB_DIM = 2048
    embs = {}
    with open('cnn_embeddings.txt') as f:
        for line in f:
            p, t, s, emb = line.strip().split('\t')
            embs[(p, t, int(s))] = np.array([float(e) for e in emb.split()])
    sys.stderr.write("Finished loading\n")
    N = len(D)
    feats = []
    lbls = []
    meta = []
    nerrs = [0, 0]
    for i, d in enumerate(D):
        sidx = int(d['step'])
        start = sidx
        end = sidx+window_size
        L = 1+end-start
        w = d['lemma']+'_'+d['pos']
        p = d['participant']
        t = d["task"]
        fv = []
        for si, step in enumerate(range(start, end+1)):
            if (p, t, step) in embs:
                e = embs[(p, t, step)]
                fv.append(e)
        fv = np.array(fv)
        nerrs[1] += 1
        try:
            avg = fv.mean(axis=0)
            mn = fv.min(axis=0)
            mx = fv.min(axis=0)
            st = fv[0, :]
            ed = fv[-1, :]
            feats.append(np.hstack((avg, mn, mx, st, ed)))
            lbls.append(w)
            meta.append(p + ' ' + t + ' %s '%sidx + '%s-%s'%(start, end))
        except ValueError:
            #print("fv", p, t, sidx, fv.shape)
            nerrs[0] += 1
    Xmat = np.array(feats)
    print("errors: ", nerrs)
    print("feats", Xmat.shape)
    return DataMat(Xmat, lbls, meta)



