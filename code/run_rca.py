# -*- coding: utf-8 -*-
"""
Root cause analysis for multistage manufacturing processes.

The diagnostic settings below are fixed and correspond to the configuration
reported in the manuscript. No parameter is selected on the test partition.

Configuration:
- The primary branch searches the event-dependent graph with the tightened
  target prior, maximum aggregation over the nodes of a unit, search depth 6,
  and a unit-repeat penalty of 0.2.
- Two action-sensitive branches are added: the steady-state dependency
  structure with the relaxed target prior and top-2 aggregation, and the
  event-dependent graph with the relaxed target prior and maximum aggregation.
- The action-adaptive gate raises the weight of the two action-sensitive
  branches when the recent standardised deviation of the monitored control
  variables is large.
- The reconstructed path uses the steady-state structure, the target prior,
  search depth 6, the repeat penalty, and near-optimal path selection.

Triggering:
- fixed post-fault horizon by default, or the validation-calibrated control
  limit with --trigger_mode alarm.
"""
from pathlib import Path
_RELEASE_ROOT = Path(__file__).resolve().parents[1]
import argparse, glob, json, math, os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

import utils as util
from engine import Trainer
import tep_meta as meta

EPS = 1e-12
FEATURES = ["boundary","dev5","support","depth","onset","change","action"]

# Evidence-fusion weight vectors.
W_E1 = np.array([
    0.05625512, 0.03390394, 0.15810331, 0.34902648,
    0.36350057, 0.02840769, 0.01080288
], dtype=float)
W_E3 = np.array([
    0.07321828, 0.01576054, 0.02276859, 0.44056191,
    0.03542811, 0.20096752, 0.21129505
], dtype=float)
W_E4 = np.array([
    0.04800971, 0.02244002, 0.07144867, 0.51984408,
    0.04294557, 0.14286900, 0.15244295
], dtype=float)

RCA_SETTINGS = dict(
    target_prior="lag025",
    target_gamma_root_main=0.1,
    target_gamma_action=1.0,
    depth=6,
    repeat_penalty=0.2,
    beam_width=20,
    action_boost=0.5,
    action_threshold=1.5,
    path_backbone="static",
    path_strategy="short_nearbest",
)

def rank01(x):
    x = np.asarray(x, float)
    if len(x) <= 1:
        return np.ones_like(x)
    return pd.Series(x).rank(method="average", pct=True).to_numpy(float)

def norm_rows(A):
    A = np.asarray(A, float)
    d = np.abs(A).sum(-1, keepdims=True)
    d[d < EPS] = 1.0
    return A / d

def smooth_prior(p, gamma):
    p = np.clip(np.asarray(p, float), 0, None)
    q = np.ones_like(p) if gamma == 0 else np.power(p + EPS, float(gamma))
    return q / (q.sum() + EPS)

def apply_target_gate(Tseq, gamma):
    T = np.asarray(Tseq, float).copy()
    g = np.ones(T.shape[-1], float)
    for n in range(len(g)):
        if meta.unit_of_node(n) not in {"PURGE","SEPARATOR"}:
            g[n] = float(gamma)
    T *= g[None, :]
    den = T.sum(axis=1, keepdims=True)
    bad = den[:,0] < EPS
    if bad.any():
        allowed = np.array(
            [meta.unit_of_node(n) in {"PURGE","SEPARATOR"} for n in range(len(g))],
            float,
        )
        allowed /= allowed.sum() + EPS
        T[bad] = allowed
        den = T.sum(axis=1, keepdims=True)
    return T / (den + EPS)

def infer_arch(state, args):
    layer_ids = {
        int(k.split(".")[1]) for k in state if k.startswith("st_blocks.")
    }
    if layer_ids:
        args.num_layers = max(layer_ids) + 1
    if "input_proj.weight" in state:
        args.hidden_dim = int(state["input_proj.weight"].shape[0])
        args.node_features = int(state["input_proj.weight"].shape[1])
    if "graph_gen.emb_linear.weight" in state:
        args.action_dim = int(state["graph_gen.emb_linear.weight"].shape[1])
    if "self_ar_raw" in state:
        args.num_nodes = int(state["self_ar_raw"].numel())
    required = ["target_head.query","graph_gen.static_scale_raw","self_ar_raw"]
    miss = [k for k in required if k not in state]
    if miss:
        raise RuntimeError("Not an AD-STGN v3 checkpoint; missing: " + ", ".join(miss))

def load_priors(args):
    tr = np.load(os.path.join(args.data_dir, "train.npz"))
    va = np.load(os.path.join(args.data_dir, "val.npz"))
    xtr, utr, ytr = tr["x"], tr["u"], tr["y"]

    mx = xtr.mean(axis=(0,1), keepdims=True)
    sx = xtr.std(axis=(0,1), keepdims=True); sx[sx == 0] = 1.0
    mu = utr.mean(axis=(0,1), keepdims=True)
    su = utr.std(axis=(0,1), keepdims=True); su[su == 0] = 1.0
    my, sy = float(ytr.mean()), float(ytr.std())
    if sy == 0: sy = 1.0

    sc_x = util.StandardScaler(mx, sx)
    sc_u = util.StandardScaler(mu, su)
    sc_y = util.StandardScaler(my, sy)

    A = np.load(
        os.path.join(args.data_dir, "A_static.npy")
    )[:args.num_nodes, :args.num_nodes].astype(np.float32)
    M = np.ones((args.num_nodes, args.num_nodes), np.float32)
    np.fill_diagonal(M, 0.0)

    Xlast = xtr[:, -1, :, 0]
    x_mu, x_sd = Xlast.mean(0), Xlast.std(0)
    x_sd[x_sd < 1e-4] = 1e-4
    Z = np.abs(Xlast - x_mu[None,:]) / x_sd[None,:]
    root_ref = np.quantile(Z, args.root_ref_quantile, axis=0)
    root_ref[root_ref < 1e-6] = 1.0

    yflat = ytr.reshape(-1)
    y_ref = float(np.quantile(np.abs(yflat-my)/sy, args.root_ref_quantile))
    if y_ref < 1e-6: y_ref = 1.0

    return dict(
        sc_x=sc_x, sc_u=sc_u, sc_y=sc_y, A=A, M=M,
        x_mu=x_mu, x_sd=x_sd, root_ref=root_ref,
        y_mu=my, y_sd=sy, y_ref=y_ref, val=va,
    )

def batched_residuals(model, x, u, y, M_t, A_t, device, batch_size):
    out = np.empty(len(x), np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            e = min(s + batch_size, len(x))
            tx = torch.FloatTensor(x[s:e]).to(device)
            tu = torch.FloatTensor(u[s:e]).to(device)
            ty = torch.FloatTensor(y[s:e])[:, -1, :].to(device)
            pred, _, _ = model(tx, tu, M_t, A_t)
            out[s:e] = torch.abs(pred - ty).mean(-1).cpu().numpy()
    return out

def compute_ucl(model, pri, args, M_t, A_t, device):
    va = pri["val"]
    x = pri["sc_x"].transform(va["x"])
    u = pri["sc_u"].transform(va["u"])
    y = pri["sc_y"].transform(va["y"])
    if args.ucl_sample_size > 0 and len(x) > args.ucl_sample_size:
        idx = np.random.default_rng(0).choice(
            len(x), args.ucl_sample_size, replace=False
        )
        x, u, y = x[idx], u[idx], y[idx]
    r = batched_residuals(
        model, x, u, y, M_t, A_t, device, args.eval_batch_size
    )
    mu, sd = float(r.mean()), float(r.std())
    return mu + args.sigma_coef*sd, mu, sd

def build_context(model, x_scaled, u_scaled, y_raw, alarm,
                  pri, args, M_t, A_t, device):
    hs = max(0, alarm - args.rca_history_steps + 1)
    idx = np.arange(hs, alarm+1)
    Aall, Tall = [], []
    with torch.no_grad():
        for s in range(0, len(idx), args.eval_batch_size):
            ids = idx[s:s+args.eval_batch_size]
            tx = torch.FloatTensor(x_scaled[ids]).to(device)
            tu = torch.FloatTensor(u_scaled[ids]).to(device)
            _, A, ta = model(tx, tu, M_t, A_t)
            Aall.append(A[:, -1].cpu().numpy())
            Tall.append(ta.cpu().numpy())
    Aseq = np.concatenate(Aall, 0)
    Tseq = np.concatenate(Tall, 0)
    Xraw = pri["sc_x"].inverse_transform(
        x_scaled[hs:alarm+1]
    )[:, -1, :, 0]
    Uraw = pri["sc_u"].inverse_transform(
        u_scaled[hs:alarm+1]
    )[:, -1, :]
    Yraw = np.asarray(y_raw[hs:alarm+1]).reshape(-1)
    return hs, Aseq, Tseq, Xraw, Uraw, Yraw

def zscore_matrix(Xraw, mu, sd, delta_max=10.0):
    z = np.abs(Xraw - mu[None,:]) / (sd[None,:] + EPS)
    return np.clip(z, 0.0, delta_max)

def boundary_scores(finals, probs, Z, root_ref):
    score = {}
    ref = np.asarray(root_ref, float)
    for c, pp in zip(finals, probs):
        pts = c["path"]
        a = [
            float(Z[int(t), int(n)]) / max(float(ref[int(n)]), EPS)
            for n,t in pts
        ]
        for i,(n,t) in enumerate(pts):
            upstream = a[i+1] if i+1 < len(a) else 0.0
            b = a[i] / (1.0 + upstream)
            score[int(n)] = score.get(int(n),0.0) + float(pp)*max(b,0.0)
    return score

def first_persistent_onset(x, patience=3):
    hit = np.asarray(x,float) >= 1.0
    run = 0
    for i,h in enumerate(hit):
        run = run+1 if h else 0
        if run >= patience:
            return int(i-patience+1)
    return None

def onset_times(Xraw, mu, sd, root_ref, patience=3):
    Z = zscore_matrix(Xraw, mu, sd)
    ref = np.asarray(root_ref,float)
    out = {}
    for n in range(Xraw.shape[1]):
        t = first_persistent_onset(
            Z[:,n] / max(ref[n],EPS), patience=patience
        )
        if t is not None:
            out[int(n)] = int(t)
    return out

def action_scores(Uraw, pri, target_time):
    mu = np.asarray(pri["sc_u"].mean).reshape(-1)
    sd = np.asarray(pri["sc_u"].std).reshape(-1)
    sd[sd < EPS] = 1.0
    a = max(0, int(target_time)-4)
    z = np.abs(Uraw[a:target_time+1]-mu[None,:]) / sd[None,:]
    au = z.max(0)
    us = {u:0.0 for u in meta.UNITS}
    for j,val in enumerate(au,1):
        u = meta.UNIT_OF_XMV.get(j)
        if u is not None:
            us[u] = max(us[u], float(val))
    arr = np.array([us[u] for u in meta.UNITS],float)
    rr = rank01(arr)
    return {u:rr[i] for i,u in enumerate(meta.UNITS)}

def action_strength(Uraw, pri, tt):
    mu = np.asarray(pri["sc_u"].mean).reshape(-1)
    sd = np.asarray(pri["sc_u"].std).reshape(-1)
    sd[sd < EPS] = 1.0
    a = max(0, int(tt)-4)
    z = np.abs(Uraw[a:tt+1]-mu[None,:]) / sd[None,:]
    return float(z.max()) if z.size else 0.0

def search_paths_complete(Gseq, Tseq, beam_width, max_depth, repeat_penalty):
    T,N,_ = Gseq.shape
    tt = T-1
    first = np.clip(np.asarray(Tseq[tt],float),0,None)
    first /= first.sum()+EPS
    beams = []
    for n in np.argsort(first)[::-1]:
        p = float(first[n])
        if p <= 0: continue
        beams.append(dict(
            path=[(int(n),tt)],
            score=math.log(p+EPS),
            logs=[math.log(p+EPS)],
        ))
    beams = beams[:int(beam_width)]
    completed = []

    for _ in range(max(0,int(max_depth)-1)):
        cands = []
        for bm in beams:
            cn,ct = bm["path"][-1]
            pt = int(ct)-1
            if pt < 0:
                completed.append(bm); continue
            probs = np.abs(np.asarray(Gseq[int(ct),int(cn),:],float)).copy()
            probs[int(cn)] = 0.0
            if probs.sum() > EPS:
                probs /= probs.sum()
            used_nodes = {int(n) for n,_ in bm["path"]}
            used_units = {meta.unit_of_node(n) for n,_ in bm["path"]}
            added = False
            for pn in np.argsort(probs)[::-1]:
                ep = float(probs[pn])
                if ep <= 0: break
                if int(pn) in used_nodes: continue
                pu = meta.unit_of_node(pn)
                penalty = float(repeat_penalty) if pu in used_units else 1.0
                if penalty <= 0: continue
                lg = math.log(ep*penalty + EPS)
                cands.append(dict(
                    path=bm["path"]+[(int(pn),pt)],
                    score=bm["score"]+lg,
                    logs=bm["logs"]+[lg],
                ))
                added = True
            if not added:
                completed.append(bm)
        if not cands:
            beams=[]; break
        cands.sort(key=lambda x:x["score"], reverse=True)
        chosen, keys = [], set()
        for c in cands:
            k = tuple(int(n) for n,_ in c["path"][:2])
            if k not in keys:
                chosen.append(c); keys.add(k)
            if len(chosen) >= int(beam_width): break
        if len(chosen) < int(beam_width):
            seen = {tuple(c["path"]) for c in chosen}
            for c in cands:
                if tuple(c["path"]) in seen: continue
                chosen.append(c)
                if len(chosen) >= int(beam_width): break
        beams = chosen

    finals = completed + beams
    uniq = {}
    for c in finals:
        cc = dict(c)
        cc["norm"] = cc["score"]/max(len(cc["logs"]),1)
        k = tuple(cc["path"])
        if k not in uniq or cc["norm"] > uniq[k]["norm"]:
            uniq[k] = cc
    finals = sorted(
        uniq.values(), key=lambda c:c["norm"], reverse=True
    )[:int(beam_width)]
    if not finals:
        raise RuntimeError("No RCA candidate paths.")
    vals = np.asarray([c["norm"] for c in finals],float)
    probs = np.exp(vals-vals.max()); probs /= probs.sum()+EPS
    return finals, probs

def node_features(records, probs, Xraw, mu, sd, root_ref, tt, patience):
    paths = [r["path"] for r in records]
    nodes = sorted({int(n) for p in paths for n,_ in p})
    Z = zscore_matrix(Xraw, mu, sd)
    ratio = Z / np.maximum(np.asarray(root_ref,float)[None,:], EPS)
    bscore = boundary_scores(records, probs, Z, root_ref)

    support=defaultdict(float); depth=defaultdict(float); dden=defaultdict(float)
    for p,pp in zip(paths,probs):
        seen=set(); L=max(len(p)-1,1)
        for i,(n,t) in enumerate(p):
            n=int(n)
            if n not in seen:
                support[n]+=float(pp); seen.add(n)
            depth[n]+=float(pp)*(i/L); dden[n]+=float(pp)

    onsets = onset_times(Xraw,mu,sd,root_ref,patience=patience)
    a5=max(0,tt-4); dev5=ratio[a5:tt+1].max(0)
    a30=max(0,tt-29); rr=ratio[a30:tt+1]
    change=np.maximum(np.diff(rr,axis=0),0).max(0) if len(rr)>=2 else np.zeros(ratio.shape[1])

    raw=np.column_stack([
        [bscore.get(n,0) for n in nodes],
        [dev5[n] for n in nodes],
        [support.get(n,0) for n in nodes],
        [depth[n]/max(dden[n],EPS) for n in nodes],
        [tt-onsets[n]+1 if n in onsets else 0 for n in nodes],
        [change[n] for n in nodes],
    ])
    F=np.column_stack([rank01(raw[:,j]) for j in range(raw.shape[1])])
    return nodes,F,onsets

def aggregate_units(nodes, F, action, mode):
    by={u:[] for u in meta.UNITS}
    for i,n in enumerate(nodes):
        by[meta.unit_of_node(n)].append(F[i])
    UF=[]
    for u in meta.UNITS:
        arr=np.asarray(by[u],float)
        if len(arr)==0:
            v=np.zeros(6)
        elif mode=="max":
            v=arr.max(0)
        elif mode=="top2":
            v=np.sort(arr,axis=0)[-min(2,len(arr)):].mean(0)
        else:
            raise ValueError(mode)
        UF.append(np.r_[v, action.get(u,0.0)])
    UF=np.asarray(UF,float)
    return np.column_stack([rank01(UF[:,j]) for j in range(UF.shape[1])])

def unit_metrics(order, gt):
    r=[meta.UNITS[int(i)] for i in order]
    top1=int(gt in r[:1]); top3=int(gt in r[:3]); top5=int(gt in r[:5])
    rr=0.0
    for i,u in enumerate(r,1):
        if u==gt:
            rr=1.0/i; break
    return top1,top3,top5,rr

def fused_root_score(s1,s3,s4,action_top):
    # Evidence fusion: base weight 1 for the primary branch and +0.25*g for the two action-sensitive branches.
    g = 1.0 / (1.0 + math.exp(-(float(action_top)-1.5)))
    w = np.array([1.0, 0.25*g, 0.25*g],float)
    w /= w.sum()+EPS
    S=np.vstack([rank01(s1),rank01(s3),rank01(s4)])
    return np.einsum("e,eu->u",w,S),g,w

def path_unit_seq_forward(path):
    seq=[]
    for n,_ in reversed(path):
        u=meta.unit_of_node(n)
        if not seq or seq[-1]!=u:
            seq.append(u)
    return seq

def truncate_record(rec, root_unit):
    idx=None
    for i,(n,t) in enumerate(rec["path"]):
        if meta.unit_of_node(n)==root_unit:
            idx=i; break
    if idx is None: return None
    p=list(rec["path"][:idx+1])
    logs=list(rec["logs"][:idx+1])
    units=path_unit_seq_forward(p)
    return dict(
        path=p, logs=logs, units=units,
        repeat_count=max(0,len(units)-len(set(units))),
        phys_edges=max(0,len(units)-1),
        avglog=float(np.mean(logs)),
    )

def select_short_nearbest(records, probs, root_unit):
    cand=[]
    for rec,pp in zip(records,probs):
        x=truncate_record(rec,root_unit)
        if x is not None:
            x["beam_prob"]=float(pp); cand.append(x)
    if not cand: return None
    best=max(x["avglog"] for x in cand)
    near=[x for x in cand if x["avglog"] >= best-math.log(2.0)]
    return min(
        near,
        key=lambda x:(x["phys_edges"]+2*x["repeat_count"],-x["avglog"])
    )["path"]

def unit_seq_with_target(path):
    if path is None: return []
    seq=path_unit_seq_forward(path)
    if not seq or seq[-1]!="PURGE":
        seq.append("PURGE")
    return seq

def edge_set(seq):
    return {(a,b) for a,b in zip(seq[:-1],seq[1:]) if a!=b}

def lcs_ratio(a,b):
    if not a or not b: return 0.0
    dp=np.zeros((len(a)+1,len(b)+1),np.int16)
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            if a[i-1]==b[j-1]:
                dp[i,j]=dp[i-1,j-1]+1
            else:
                dp[i,j]=max(dp[i-1,j],dp[i,j-1])
    return float(dp[-1,-1]/len(b))

def physical_metrics(path, gt):
    if path is None or gt not in meta.PRIMARY_REF:
        return dict(
            physical_edge_p=0.0,physical_edge_r=0.0,physical_edge_f1=0.0,
            physical_recovery=0.0,topology_precision=0.0,physical_path=""
        )
    pred=unit_seq_with_target(path); ref=meta.PRIMARY_REF[gt]
    pe,re=edge_set(pred),edge_set(ref)
    tp=len(pe&re)
    p=tp/max(len(pe),1); r=tp/max(len(re),1)
    f=2*p*r/(p+r) if p+r else 0.0
    topo=sum(e in meta.ALLOWED_UNIT_EDGES for e in pe)/max(len(pe),1)
    return dict(
        physical_edge_p=float(p),physical_edge_r=float(r),
        physical_edge_f1=float(f),physical_recovery=lcs_ratio(pred,ref),
        topology_precision=float(topo),physical_path="->".join(pred)
    )

def empirical_lag(x_up,x_dn,max_lag):
    xu=(x_up-x_up.mean())/(x_up.std()+1e-8)
    xd=(x_dn-x_dn.mean())/(x_dn.std()+1e-8)
    best,bk=-np.inf,0
    for k in range(max_lag+1):
        if k==0:
            c=float(np.dot(xu,xd)/len(xd))
        else:
            if len(xd)<=k: break
            c=float(np.dot(xu[:-k],xd[k:])/len(xd[k:]))
        if abs(c)>best:
            best,bk=abs(c),k
    return bk

def delay_metric(path,onsets,Xraw,max_lag):
    if not path: return np.nan,0
    chron=list(reversed(path))
    errs=[]
    for (up,_),(dn,_) in zip(chron[:-1],chron[1:]):
        up,dn=int(up),int(dn)
        if meta.unit_of_node(up)==meta.unit_of_node(dn): continue
        if up not in onsets or dn not in onsets: continue
        pred=max(0,int(onsets[dn])-int(onsets[up]))
        emp=empirical_lag(Xraw[:,up],Xraw[:,dn],max_lag)
        errs.append(abs(pred-emp))
    return (float(np.mean(errs)) if errs else np.nan),len(errs)

def evaluate_event_file(fp, model, pri, args, Mt, At, device, UCL=None):
    d=np.load(fp,allow_pickle=True)
    fid=int(d["fault_id"]); eid=int(d["event_id"])
    if fid not in meta.FAULT_GT: return None
    gt=meta.FAULT_GT[fid]["gt_unit"]
    fs=int(d["fault_start"])

    x=pri["sc_x"].transform(d["x"])
    u=pri["sc_u"].transform(d["u"])
    ys=pri["sc_y"].transform(d["y"])
    yr=d["y"]

    if args.trigger_mode=="fixed":
        alarm=min(fs+args.fixed_delay,len(x)-1)
        detected=1
    else:
        offset=max(0,fs-args.patience+1)
        loss=batched_residuals(
            model,x[offset:],u[offset:],ys[offset:],
            Mt,At,device,args.eval_batch_size
        )
        alarm=None
        for iloc in range(len(loss)):
            gi=offset+iloc
            if gi<fs or iloc<args.patience-1: continue
            w=loss[iloc-args.patience+1:iloc+1]
            hit=(np.all(w>UCL) if args.alarm_rule=="consecutive"
                 else (loss[iloc]>UCL and np.mean(w)>UCL))
            if hit:
                alarm=gi; break
        if alarm is None:
            return dict(
                fault_id=fid,event_id=eid,gt_unit=gt,detected=0,
                trigger_delay=np.nan,unit_top1=0,unit_top3=0,unit_top5=0,
                unit_mrr=0.0,physical_edge_p=0.0,physical_edge_r=0.0,
                physical_edge_f1=0.0,physical_recovery=0.0,
                topology_precision=0.0,root_path_coverage=0,
                source_anybeam=0,delay_mae=np.nan,delay_edges=0,
                root_time_error=np.nan,predicted_root_unit="",
                physical_path=""
            )
        detected=1

    hs,Adyn,Latt,Xraw,Uraw,Yraw=build_context(
        model,x,u,yr,alarm,pri,args,Mt,At,device
    )
    tt=len(Xraw)-1
    lag=np.load(os.path.join(args.data_dir,"target_lag_prior.npy")).astype(float)
    lag/=lag.sum()+EPS
    lag025=np.repeat(smooth_prior(lag,0.25)[None,:],len(Adyn),axis=0)
    Tsoft=apply_target_gate(lag025,0.1)
    Topen=apply_target_gate(lag025,1.0)

    Gdyn=norm_rows(Adyn)
    Gsta=np.repeat(norm_rows(pri["A"])[None,:,:],len(Adyn),axis=0)
    act=action_scores(Uraw,pri,tt)

    # Dynamic/onset branch
    rec1,p1=search_paths_complete(
        Gdyn,Tsoft,args.beam_width,6,0.2
    )
    n1,F1,o1=node_features(
        rec1,p1,Xraw,pri["x_mu"],pri["x_sd"],pri["root_ref"],
        tt,args.onset_patience
    )
    UF1=aggregate_units(n1,F1,act,"max")
    s1=UF1@W_E1

    # Static/action branch
    rec3,p3=search_paths_complete(
        Gsta,Topen,args.beam_width,6,0.2
    )
    n3,F3,o3=node_features(
        rec3,p3,Xraw,pri["x_mu"],pri["x_sd"],pri["root_ref"],
        tt,args.onset_patience
    )
    UF3=aggregate_units(n3,F3,act,"top2")
    s3=UF3@W_E3

    # Dynamic/action branch
    rec4,p4=search_paths_complete(
        Gdyn,Topen,args.beam_width,6,0.2
    )
    n4,F4,o4=node_features(
        rec4,p4,Xraw,pri["x_mu"],pri["x_sd"],pri["root_ref"],
        tt,args.onset_patience
    )
    UF4=aggregate_units(n4,F4,act,"max")
    s4=UF4@W_E4

    atop=action_strength(Uraw,pri,tt)
    score,gate,effw=fused_root_score(s1,s3,s4,atop)
    order=np.argsort(score)[::-1]
    pred_root=meta.UNITS[int(order[0])]
    t1,t3,t5,mrr=unit_metrics(order,gt)

    # Path backbone: static graph with the target-relevance gate.
    recp,pp=search_paths_complete(
        Gsta,Tsoft,args.beam_width,6,0.2
    )
    path=select_short_nearbest(recp,pp,pred_root)
    pm=physical_metrics(path,gt)

    candidate_units={
        meta.unit_of_node(n)
        for r in recp for n,_ in r["path"]
    }
    source_any=int(gt in candidate_units)

    onsets=onset_times(
        Xraw,pri["x_mu"],pri["x_sd"],pri["root_ref"],
        patience=args.onset_patience
    )
    dmae,dn=delay_metric(path,onsets,Xraw,args.delay_eval_max_lag)

    root_time_error=np.nan
    if path:
        root_node=int(path[-1][0])
        if root_node in onsets:
            root_abs=hs+int(onsets[root_node])
            root_time_error=abs(root_abs-fs)

    return dict(
        fault_id=fid,event_id=eid,gt_unit=gt,detected=detected,
        trigger_delay=int(alarm-fs),
        predicted_root_unit=pred_root,
        unit_top1=t1,unit_top3=t3,unit_top5=t5,unit_mrr=mrr,
        root_path_coverage=int(path is not None),
        source_anybeam=source_any,
        action_top_z=atop,action_gate=gate,
        effective_w_E1=float(effw[0]),
        effective_w_E3=float(effw[1]),
        effective_w_E4=float(effw[2]),
        delay_mae=dmae,delay_edges=dn,
        root_time_error=root_time_error,
        **pm
    )

def summarize(df):
    metrics=[
        "unit_top1","unit_top3","unit_top5","unit_mrr",
        "physical_edge_p","physical_edge_r","physical_edge_f1",
        "physical_recovery","topology_precision","root_path_coverage",
        "source_anybeam","delay_mae","root_time_error"
    ]
    micro=df[metrics].mean(numeric_only=True)
    macro=(
        df.groupby("gt_unit")[metrics].mean(numeric_only=True)
          .mean(numeric_only=True)
    )
    return pd.DataFrame({
        "micro":micro,
        "macro_by_root_unit":macro,
    })

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        default=str(_RELEASE_ROOT / 'data/processed_tep')
    )
    ap.add_argument("--model_path",default=None)
    ap.add_argument("--event_prefix",default="tep")
    ap.add_argument("--out_dir",default=None)
    ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--trigger_mode",choices=["fixed","alarm"],default="fixed")
    ap.add_argument("--fixed_delay",type=int,default=60)
    ap.add_argument("--beam_width",type=int,default=20)
    ap.add_argument("--rca_history_steps",type=int,default=240)
    ap.add_argument("--onset_patience",type=int,default=3)
    ap.add_argument("--root_ref_quantile",type=float,default=.99)
    ap.add_argument("--delay_eval_max_lag",type=int,default=15)
    ap.add_argument("--eval_batch_size",type=int,default=256)
    ap.add_argument("--sigma_coef",type=float,default=3.0)
    ap.add_argument("--patience",type=int,default=5)
    ap.add_argument("--alarm_rule",choices=["moving_average","consecutive"],default="moving_average")
    ap.add_argument("--ucl_sample_size",type=int,default=5000)
    ap.add_argument("--num_nodes",type=int,default=40)
    ap.add_argument("--action_dim",type=int,default=11)
    ap.add_argument("--node_features",type=int,default=1)
    ap.add_argument("--hidden_dim",type=int,default=64)
    ap.add_argument("--num_layers",type=int,default=3)
    ap.add_argument("--out_dim",type=int,default=1)
    args=ap.parse_args()

    if args.model_path is None:
        args.model_path=os.path.join(args.data_dir,"rca_model.pth")
    if args.out_dir is None:
        args.out_dir=os.path.join(
            os.path.dirname(args.data_dir),
            f"rca_{args.event_prefix}"
        )
    os.makedirs(args.out_dir,exist_ok=True)

    required=[
        "train.npz","val.npz","A_static.npy",
        "target_lag_prior.npy"
    ]
    miss=[x for x in required if not os.path.exists(os.path.join(args.data_dir,x))]
    if not os.path.exists(args.model_path):
        miss.append(args.model_path)
    if miss:
        raise FileNotFoundError("Missing required files: "+", ".join(miss))

    state=torch.load(args.model_path,map_location="cpu")
    infer_arch(state,args)
    device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    pri=load_priors(args)
    At=torch.FloatTensor(pri["A"]).to(device)
    Mt=torch.FloatTensor(pri["M"]).to(device)

    eng=Trainer(
        pri["sc_y"],args.node_features,args.action_dim,args.hidden_dim,
        args.num_nodes,args.num_layers,args.out_dim,
        1e-3,1e-4,device,.1,.02
    )
    eng.model.load_state_dict(state)
    eng.model.eval()

    UCL=None
    if args.trigger_mode=="alarm":
        UCL,mu,sd=compute_ucl(eng.model,pri,args,Mt,At,device)
        print(f"UCL={UCL:.6f} (mu={mu:.6f}, sd={sd:.6f})")

    files=sorted(glob.glob(
        os.path.join(args.data_dir,f"{args.event_prefix}_f*_e*.npz")
    ))
    files=[
        fp for fp in files
        if int(np.load(fp,allow_pickle=True)["fault_id"]) in meta.FAULT_GT
    ]
    if not files:
        raise FileNotFoundError(
            f"No event files matching {args.event_prefix}_f*_e*.npz"
        )

    print("AD-STGN root-cause analysis")
    print("Device:",device)
    print("Events:",len(files))
    print("Trigger:",args.trigger_mode)
    print("Reported configuration.")
    rows=[]
    for i,fp in enumerate(files,1):
        r=evaluate_event_file(
            fp,eng.model,pri,args,Mt,At,device,UCL
        )
        rows.append(r)
        print(f"[{i}/{len(files)}] {os.path.basename(fp)}")

    df=pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir,"rca_event_results.csv"),index=False)
    byf=df.groupby(["fault_id","gt_unit"])[[
        "unit_top1","unit_top3","unit_top5","unit_mrr",
        "physical_edge_p","physical_edge_r","physical_edge_f1",
        "physical_recovery","topology_precision","root_path_coverage",
        "source_anybeam","delay_mae","root_time_error"
    ]].mean(numeric_only=True)
    byf.to_csv(os.path.join(args.out_dir,"rca_by_fault.csv"))
    sm=summarize(df)
    sm.to_csv(os.path.join(args.out_dir,"rca_summary.csv"))

    audit=dict(
        version="AD-STGN",
        settings=RCA_SETTINGS,
        recipes=dict(
            rand_080=W_E1.tolist(),
            rand_048=W_E3.tolist(),
            rand_052=W_E4.tolist(),
        ),
        primary_reference_paths=meta.PRIMARY_REF,
        note="Settings fixed for the reported evaluation."
    )
    with open(
        os.path.join(args.out_dir,"config_audit.json"),
        "w",encoding="utf-8"
    ) as f:
        json.dump(audit,f,ensure_ascii=False,indent=2)

    print("\n===== FINAL summary =====")
    print(sm.to_string())
    if args.trigger_mode=="alarm":
        print(f"\nDetection coverage={df.detected.mean():.4f}")

if __name__=="__main__":
    main()
