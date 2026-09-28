# -*- coding: utf-8 -*-
r"""
Monitoring and root cause analysis.

The script loads the Soft-Delay predictor, applies the statistical process
control alarm rule, and calls the root cause analysis module for every alarm.
No parameter is selected on the test partition.

Monitoring
----------
- checkpoint candidates: monitor_run1.pth ... monitor_runN.pth
- formal architecture is inferred from each checkpoint; only tau_max=24,
  num_layers=1 checkpoints are eligible
- the eligible checkpoint with the LOWEST validation mean absolute residual
  is selected before the test evaluation
- predictor static prior:
    A_static_predictor.npy
- lag prior:
    tau_prior.npy
- original SPC rule:
    e_t > UCL  AND  mean(e_{t-4:t}) > UCL
    UCL = mean(validation residual) + 3 * std(validation residual)

RCA side
--------
Uses the run_rca.py implementation and
rca_model.pth. No diagnostic setting is modified.

Outputs
-------
checkpoint_validation.csv
monitor_event_results.csv
monitor_by_fault.csv
monitor_summary.csv
config_audit.json
"""

from pathlib import Path
_RELEASE_ROOT = Path(__file__).resolve().parents[1]
import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os

import numpy as np
import pandas as pd
import torch

import run_rca as F
import tep_meta as meta
from engine import Trainer

EPS = 1e-12

def sha256(path, block=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def load_soft_delay_model(model_file):
    p = os.path.abspath(model_file)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location("model_soft_delay", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.AD_STGN

def infer_soft_delay_arch(state):
    keys = set(state.keys())
    if "st_blocks.0.phi" not in keys:
        raise RuntimeError("Checkpoint has no phi/psi Soft-Delay parameters.")
    n, tau = tuple(state["st_blocks.0.phi"].shape)
    layers = sorted({
        int(k.split(".")[1])
        for k in keys
        if k.startswith("st_blocks.")
        and len(k.split(".")) > 2
        and k.split(".")[1].isdigit()
    })
    hidden = int(state["input_proj.weight"].shape[0])
    node_features = int(state["input_proj.weight"].shape[1])
    action_dim = int(state["graph_gen.emb_linear.weight"].shape[1])
    # Original TEP Soft-Delay predictor has one scalar quality target.
    # Do not infer this from an old checkpoint head key name, because older
    # saved runs may use a different output-head key while the architecture
    # is otherwise identical.
    out_dim = 1
    return dict(
        num_nodes=int(n),
        tau_max=int(tau),
        num_layers=len(layers),
        hidden_dim=hidden,
        node_features=node_features,
        action_dim=action_dim,
        out_dim=out_dim,
    )

def build_delta_prior(tau_prior, tau_max, alpha=0.05):
    tau_prior = np.asarray(tau_prior)
    n = tau_prior.shape[0]
    d = np.zeros((n, n, tau_max), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            delay = int(tau_prior[i, j])
            delay = min(max(delay, 0), tau_max - 1)
            if delay > 0:
                d[i, j, delay] = float(alpha)
    return d

def load_scalers(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"))
    xtr = np.asarray(tr["x"], np.float32)
    utr = np.asarray(tr["u"], np.float32)
    ytr = np.asarray(tr["y"], np.float32)

    mx = xtr.mean(axis=(0, 1), keepdims=True)
    sx = xtr.std(axis=(0, 1), keepdims=True)
    sx[sx == 0] = 1.0

    mu = utr.mean(axis=(0, 1), keepdims=True)
    su = utr.std(axis=(0, 1), keepdims=True)
    su[su == 0] = 1.0

    my = float(ytr.mean())
    sy = float(ytr.std())
    if sy == 0:
        sy = 1.0

    return dict(mx=mx, sx=sx, mu=mu, su=su, my=my, sy=sy)

def scale_x(x, sc):
    return (np.asarray(x, np.float32) - sc["mx"]) / sc["sx"]

def scale_u(u, sc):
    return (np.asarray(u, np.float32) - sc["mu"]) / sc["su"]

def scale_y(y, sc):
    return (np.asarray(y, np.float32) - sc["my"]) / sc["sy"]

def soft_delay_residuals(model, x, u, y, M_t, A_t, d_t, device, batch_size):
    out = np.empty(len(x), np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            e = min(s + batch_size, len(x))
            tx = torch.FloatTensor(x[s:e]).to(device)
            tu = torch.FloatTensor(u[s:e]).to(device)
            ty = torch.FloatTensor(y[s:e])[:, -1, :].to(device)
            pred, _ = model(tx, tu, M_t, A_t, d_t)
            out[s:e] = torch.abs(pred - ty).mean(-1).cpu().numpy()
    return out

def build_soft_delay_model(AD_STGN, ckpt, device):
    state = torch.load(ckpt, map_location="cpu")
    arch = infer_soft_delay_arch(state)
    if arch["tau_max"] != 24 or arch["num_layers"] != 1:
        raise RuntimeError(
            f"Not an eligible formal Soft-Delay checkpoint: {ckpt}; arch={arch}"
        )
    model = AD_STGN(
        node_features=arch["node_features"],
        action_dim=arch["action_dim"],
        d=arch["hidden_dim"],
        num_nodes=arch["num_nodes"],
        tau_max=arch["tau_max"],
        num_layers=arch["num_layers"],
        out_dim=arch["out_dim"],
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, arch

def select_checkpoint_by_validation(
    AD_STGN, ckpts, val, sc, M_t, A_t, d_t, device, batch_size
):
    xv = scale_x(val["x"], sc)
    uv = scale_u(val["u"], sc)
    yv = scale_y(val["y"], sc)

    rows = []
    best = None

    for p in ckpts:
        try:
            model, arch = build_soft_delay_model(AD_STGN, p, device)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"[REJECT] {os.path.basename(p)} -> {msg}")
            rows.append(dict(
                checkpoint=p,
                eligible=0,
                val_mean=np.nan,
                val_sd=np.nan,
                UCL=np.nan,
                note=msg,
            ))
            continue

        r = soft_delay_residuals(
            model, xv, uv, yv, M_t, A_t, d_t, device, batch_size
        )
        mu = float(np.mean(r))
        sd = float(np.std(r))
        ucl = mu + 3.0 * sd

        rows.append(dict(
            checkpoint=p,
            eligible=1,
            val_mean=mu,
            val_sd=sd,
            UCL=ucl,
            tau_max=arch["tau_max"],
            num_layers=arch["num_layers"],
            sha256=sha256(p),
            note="",
        ))

        if best is None or mu < best["val_mean"]:
            best = dict(
                checkpoint=p,
                model=model,
                arch=arch,
                val_mean=mu,
                val_sd=sd,
                UCL=ucl,
            )

    df = pd.DataFrame(rows)
    if best is None:
        # Caller cannot save after an exception, so emit the full rejection
        # table here before stopping.
        print("\n===== Soft-Delay checkpoint compatibility audit =====")
        if len(df):
            print(df[["checkpoint","eligible","note"]].to_string(index=False))
        raise RuntimeError(
            "No loadable tau24/layer1 Soft-Delay checkpoint was found. "
            "See the [REJECT] lines above for the exact reason."
        )

    df["selected"] = (
        df["checkpoint"].astype(str) == str(best["checkpoint"])
    ).astype(int)
    return best, df

def detect_original_spc(residuals, fault_start, ucl, patience=5):
    """
    Exact rule from the original training script:
      residual[t] > UCL
      AND mean(residual[t-4:t]) > UCL

    The FIRST alarm is used.  If it occurs before fault_start it is a false alarm
    and the event is not counted as successfully detected.
    """
    residuals = np.asarray(residuals, float)
    alarm = None
    for t in range(len(residuals)):
        if residuals[t] > ucl and t > (patience - 1):
            if float(np.mean(residuals[t-patience+1:t+1])) > ucl:
                alarm = int(t)
                break

    if alarm is None:
        return dict(
            alarm_index=np.nan,
            detected=0,
            false_alarm=0,
            no_alarm=1,
            fdd=np.nan,
        )

    if alarm < int(fault_start):
        return dict(
            alarm_index=alarm,
            detected=0,
            false_alarm=1,
            no_alarm=0,
            fdd=float(alarm - int(fault_start)),
        )

    return dict(
        alarm_index=alarm,
        detected=1,
        false_alarm=0,
        no_alarm=0,
        fdd=float(alarm - int(fault_start)),
    )

def run_rca_at_alarm(fp, alarm, model, pri, args, Mt, At, device):
    d = np.load(fp, allow_pickle=True)
    fid = int(d["fault_id"])
    eid = int(d["event_id"])
    gt = meta.FAULT_GT[fid]["gt_unit"]
    fs = int(d["fault_start"])

    x = pri["sc_x"].transform(d["x"])
    u = pri["sc_u"].transform(d["u"])
    yr = d["y"]

    hs, Adyn, Latt, Xraw, Uraw, Yraw = F.build_context(
        model, x, u, yr, int(alarm), pri, args, Mt, At, device
    )
    tt = len(Xraw) - 1

    lag = np.load(
        os.path.join(args.data_dir, "target_lag_prior.npy")
    ).astype(float)
    lag /= lag.sum() + EPS
    lag025 = np.repeat(F.smooth_prior(lag, 0.25)[None, :], len(Adyn), axis=0)
    Tsoft = F.apply_target_gate(lag025, 0.1)
    Topen = F.apply_target_gate(lag025, 1.0)

    Gdyn = F.norm_rows(Adyn)
    Gsta = np.repeat(F.norm_rows(pri["A"])[None, :, :], len(Adyn), axis=0)
    act = F.action_scores(Uraw, pri, tt)

    # Dynamic/onset branch
    rec1, p1 = F.search_paths_complete(
        Gdyn, Tsoft, args.beam_width, 6, 0.2
    )
    n1, FF1, _ = F.node_features(
        rec1, p1, Xraw, pri["x_mu"], pri["x_sd"], pri["root_ref"],
        tt, args.onset_patience
    )
    UF1 = F.aggregate_units(n1, FF1, act, "max")
    s1 = UF1 @ F.W_E1

    # Static/action branch
    rec3, p3 = F.search_paths_complete(
        Gsta, Topen, args.beam_width, 6, 0.2
    )
    n3, FF3, _ = F.node_features(
        rec3, p3, Xraw, pri["x_mu"], pri["x_sd"], pri["root_ref"],
        tt, args.onset_patience
    )
    UF3 = F.aggregate_units(n3, FF3, act, "top2")
    s3 = UF3 @ F.W_E3

    # Dynamic/action branch
    rec4, p4 = F.search_paths_complete(
        Gdyn, Topen, args.beam_width, 6, 0.2
    )
    n4, FF4, _ = F.node_features(
        rec4, p4, Xraw, pri["x_mu"], pri["x_sd"], pri["root_ref"],
        tt, args.onset_patience
    )
    UF4 = F.aggregate_units(n4, FF4, act, "max")
    s4 = UF4 @ F.W_E4

    atop = F.action_strength(Uraw, pri, tt)
    score, gate, effw = F.fused_root_score(s1, s3, s4, atop)
    order = np.argsort(score)[::-1]
    pred_root = meta.UNITS[int(order[0])]
    t1, t3, t5, mrr = F.unit_metrics(order, gt)

    # Static path backbone
    recp, pp = F.search_paths_complete(
        Gsta, Tsoft, args.beam_width, 6, 0.2
    )
    path = F.select_short_nearbest(recp, pp, pred_root)
    pm = F.physical_metrics(path, gt)

    candidate_units = {
        meta.unit_of_node(n)
        for r in recp
        for n, _ in r["path"]
    }

    return dict(
        fault_id=fid,
        event_id=eid,
        gt_unit=gt,
        predicted_root_unit=pred_root,
        unit_top1=t1,
        unit_top3=t3,
        unit_top5=t5,
        unit_mrr=mrr,
        root_path_coverage=int(path is not None),
        source_anybeam=int(gt in candidate_units),
        action_top_z=atop,
        action_gate=gate,
        effective_w_E1=float(effw[0]),
        effective_w_E3=float(effw[1]),
        effective_w_E4=float(effw[2]),
        **pm,
    )

def failed_rca_row(fid, eid, gt):
    return dict(
        fault_id=fid,
        event_id=eid,
        gt_unit=gt,
        predicted_root_unit="",
        unit_top1=0,
        unit_top3=0,
        unit_top5=0,
        unit_mrr=0.0,
        root_path_coverage=0,
        source_anybeam=0,
        action_top_z=np.nan,
        action_gate=np.nan,
        effective_w_E1=np.nan,
        effective_w_E3=np.nan,
        effective_w_E4=np.nan,
        physical_edge_p=0.0,
        physical_edge_r=0.0,
        physical_edge_f1=0.0,
        physical_recovery=0.0,
        topology_precision=0.0,
        physical_path="",
    )

def macro_by_root(df, cols):
    return (
        df.groupby("gt_unit")[cols]
        .mean(numeric_only=True)
        .mean(numeric_only=True)
    )

def summarize_hybrid(df):
    detected = df["detected"].astype(int) == 1
    det = df.loc[detected].copy()

    rca_cols = [
        "unit_top1", "unit_top3", "unit_top5", "unit_mrr",
        "physical_edge_p", "physical_edge_r", "physical_edge_f1",
        "physical_recovery",
    ]

    out = dict(
        n_total=int(len(df)),
        n_detected=int(detected.sum()),
        n_false_alarm=int(df["false_alarm"].sum()),
        n_no_alarm=int(df["no_alarm"].sum()),
        detection_coverage=float(detected.mean()),
        false_alarm_event_rate=float(df["false_alarm"].mean()),
        no_alarm_event_rate=float(df["no_alarm"].mean()),
        mean_fdd_detected=float(det["fdd"].mean()) if len(det) else np.nan,
        median_fdd_detected=float(det["fdd"].median()) if len(det) else np.nan,
    )

    if len(det):
        mi = det[rca_cols].mean(numeric_only=True)
        ma = macro_by_root(det, rca_cols)
        for c in rca_cols:
            out[f"conditional_micro_{c}"] = float(mi[c])
            out[f"conditional_macro_{c}"] = float(ma[c])
    else:
        for c in rca_cols:
            out[f"conditional_micro_{c}"] = np.nan
            out[f"conditional_macro_{c}"] = np.nan

    # End-to-end: undetected or false-alarm events count as misses.
    for c in ["unit_top1", "unit_top3", "unit_top5"]:
        out[f"end_to_end_micro_{c}"] = float(df[c].mean())

    ma_e2e = macro_by_root(
        df, ["unit_top1", "unit_top3", "unit_top5"]
    )
    for c in ["unit_top1", "unit_top3", "unit_top5"]:
        out[f"end_to_end_macro_{c}"] = float(ma_e2e[c])

    return pd.DataFrame([out])

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data_dir",
        default=str(_RELEASE_ROOT / 'data/processed_tep'),
    )
    ap.add_argument(
        "--soft_delay_model_file",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "model_soft_delay.py"
        ),
        help="Soft-Delay model definition that matches the released checkpoints.",
    )
    ap.add_argument(
        "--predictor_static",
        default=str(_RELEASE_ROOT / 'data/processed_tep/A_static_predictor.npy'),
    )
    ap.add_argument(
        "--tau_prior",
        default=str(_RELEASE_ROOT / 'data/processed_tep/tau_prior.npy'),
    )
    ap.add_argument(
        "--rca_model_path",
        default=str(_RELEASE_ROOT / 'data/processed_tep/rca_model.pth'),
    )
    ap.add_argument("--event_prefix", default="tep")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--sigma_coef", type=float, default=3.0)
    ap.add_argument("--beam_width", type=int, default=20)
    ap.add_argument("--rca_history_steps", type=int, default=240)
    ap.add_argument("--onset_patience", type=int, default=3)
    ap.add_argument("--root_ref_quantile", type=float, default=0.99)
    ap.add_argument("--delay_eval_max_lag", type=int, default=15)
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument(
        "--out_dir",
        default=str(_RELEASE_ROOT / 'results/monitor'),
    )

    # Architecture placeholders; infer_arch overwrites them.
    ap.add_argument("--num_nodes", type=int, default=40)
    ap.add_argument("--action_dim", type=int, default=11)
    ap.add_argument("--node_features", type=int, default=1)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--out_dim", type=int, default=1)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    print("AD-STGN monitoring and root-cause analysis")
    print("Device:", device)
    print("Settings fixed for the reported evaluation.")

    # Soft-Delay assets
    AD_STGN = load_soft_delay_model(args.soft_delay_model_file)
    print(
        f"Soft-Delay model definition: {args.soft_delay_model_file} | "
        f"sha256={sha256(args.soft_delay_model_file)[:16]}"
    )

    if not os.path.exists(args.predictor_static):
        raise FileNotFoundError(args.predictor_static)
    Aold_full = np.load(args.predictor_static)
    Aold = np.asarray(Aold_full[:40, :40], np.float32)
    Mold = (Aold > 0).astype(np.float32) + np.eye(40, dtype=np.float32)
    Mold = np.clip(Mold, 0, 1)

    tau_prior = np.load(args.tau_prior)[:40, :40]
    delta = build_delta_prior(tau_prior, 24, alpha=0.05)

    Aold_t = torch.FloatTensor(Aold).to(device)
    Mold_t = torch.FloatTensor(Mold).to(device)
    delta_t = torch.FloatTensor(delta).to(device)

    print(
        f"Predictor static prior: {args.predictor_static} | "
        f"sensor nonzero={np.count_nonzero(Aold)} | "
        f"sha256={sha256(args.predictor_static)[:16]}"
    )
    print(
        f"Lag prior: {args.tau_prior} | "
        f"shape={tau_prior.shape} | "
        f"sha256={sha256(args.tau_prior)[:16]}"
    )

    # Select intact formal checkpoint using VALIDATION only.
    # run0 is automatically excluded because its arch is tau15/layers3.
    ckpts = sorted(glob.glob(
        os.path.join(args.data_dir, "monitor_run*.pth")
    ))
    if not ckpts:
        raise FileNotFoundError("No monitor_run*.pth files.")

    val = np.load(os.path.join(args.data_dir, "val.npz"))
    sc = load_scalers(args.data_dir)

    best, val_df = select_checkpoint_by_validation(
        AD_STGN, ckpts, val, sc,
        Mold_t, Aold_t, delta_t,
        device, args.batch_size,
    )
    val_df.to_csv(
        os.path.join(args.out_dir, "checkpoint_validation.csv"),
        index=False,
    )

    soft_model = best["model"]
    UCL = best["val_mean"] + args.sigma_coef * best["val_sd"]

    print("\nValidation-selected Soft-Delay checkpoint:")
    print(" ", best["checkpoint"])
    print("  arch:", best["arch"])
    print(
        f"  validation residual mean={best['val_mean']:.6f}, "
        f"sd={best['val_sd']:.6f}, UCL={UCL:.6f}"
    )

    # Root cause analysis model
    state = torch.load(args.rca_model_path, map_location="cpu")
    F.infer_arch(state, args)

    pri = F.load_priors(args)
    At = torch.FloatTensor(pri["A"]).to(device)
    Mt = torch.FloatTensor(pri["M"]).to(device)

    rca_engine = Trainer(
        pri["sc_y"],
        args.node_features,
        args.action_dim,
        args.hidden_dim,
        args.num_nodes,
        args.num_layers,
        args.out_dim,
        1e-3,
        1e-4,
        device,
        0.1,
        0.02,
    )
    rca_engine.model.load_state_dict(state)
    rca_engine.model.eval()

    # Test partition
    files = sorted(glob.glob(
        os.path.join(args.data_dir, f"{args.event_prefix}_f*_e*.npz")
    ))
    files = [
        p for p in files
        if int(np.load(p, allow_pickle=True)["fault_id"]) in meta.FAULT_GT
    ]
    if len(files) != 130:
        print(f"[WARN] Expected 130 events, found {len(files)}")

    rows = []
    for ii, fp in enumerate(files, 1):
        d = np.load(fp, allow_pickle=True)
        fid = int(d["fault_id"])
        eid = int(d["event_id"])
        gt = meta.FAULT_GT[fid]["gt_unit"]
        fs = int(d["fault_start"])

        xs = scale_x(d["x"], sc)
        us = scale_u(d["u"], sc)
        ys = scale_y(d["y"], sc)

        residuals = soft_delay_residuals(
            soft_model, xs, us, ys,
            Mold_t, Aold_t, delta_t,
            device, args.batch_size,
        )

        det = detect_original_spc(
            residuals, fs, UCL, patience=args.patience
        )

        if det["detected"]:
            rca = run_rca_at_alarm(
                fp,
                int(det["alarm_index"]),
                rca_engine.model,
                pri,
                args,
                Mt,
                At,
                device,
            )
        else:
            rca = failed_rca_row(fid, eid, gt)

        row = dict(
            **rca,
            detected=int(det["detected"]),
            false_alarm=int(det["false_alarm"]),
            no_alarm=int(det["no_alarm"]),
            alarm_index=det["alarm_index"],
            fdd=det["fdd"],
            soft_ucl=UCL,
            max_prefault_residual=float(
                np.max(residuals[:fs]) if fs > 0 else np.nan
            ),
            max_postfault_residual=float(
                np.max(residuals[fs:]) if fs < len(residuals) else np.nan
            ),
        )
        rows.append(row)

        status = (
            f"DETECTED FDD={det['fdd']:.0f}"
            if det["detected"]
            else ("FALSE ALARM" if det["false_alarm"] else "NO ALARM")
        )
        print(
            f"[{ii}/{len(files)}] {os.path.basename(fp)} | {status}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(args.out_dir, "monitor_event_results.csv"),
        index=False,
    )

    metrics = [
        "detected", "false_alarm", "no_alarm", "fdd",
        "unit_top1", "unit_top3", "unit_top5", "unit_mrr",
        "physical_edge_p", "physical_edge_r",
        "physical_edge_f1", "physical_recovery",
    ]
    byf = (
        df.groupby(["fault_id", "gt_unit"])[metrics]
        .mean(numeric_only=True)
    )
    byf.to_csv(
        os.path.join(args.out_dir, "monitor_by_fault.csv")
    )

    summary = summarize_hybrid(df)
    summary.to_csv(
        os.path.join(args.out_dir, "monitor_summary.csv"),
        index=False,
    )

    audit = dict(
        method="AD-STGN monitor and root-cause analysis",
        checkpoint_selection="minimum validation mean residual among eligible run1-run9 formal checkpoints",
        selected_softdelay_checkpoint=best["checkpoint"],
        selected_softdelay_checkpoint_sha256=sha256(best["checkpoint"]),
        selected_softdelay_arch=best["arch"],
        soft_delay_model_file=args.soft_delay_model_file,
        soft_delay_model_file_sha256=sha256(args.soft_delay_model_file),
        predictor_static_path=args.predictor_static,
        predictor_static_sha256=sha256(args.predictor_static),
        predictor_static_sensor_nonzero=int(np.count_nonzero(Aold)),
        tau_prior_path=args.tau_prior,
        tau_prior_sha256=sha256(args.tau_prior),
        spc_rule="residual_t > UCL AND mean(last 5 residuals) > UCL",
        sigma_coef=args.sigma_coef,
        patience=args.patience,
        soft_ucl=UCL,
        rca_checkpoint=args.rca_model_path,
        rca_checkpoint_sha256=sha256(args.rca_model_path),
        rca_settings=F.RCA_SETTINGS,
        note="Settings fixed for the reported evaluation.",
    )
    with open(
        os.path.join(args.out_dir, "config_audit.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    print("\n===== HYBRID FINAL SUMMARY =====")
    print(summary.to_string(index=False))
    print("\nResults:", args.out_dir)

if __name__ == "__main__":
    main()
