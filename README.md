# AD-STGN

Reference implementation of **Action-Conditioned Delay-Aware Spatio-Temporal
Graph Network for Root Cause Analysis in Continuous Multistage Manufacturing
Systems**.

Jiaqi Zhang, Liangxing Shi, Zhen He. Tianjin University.

## Repository layout

| Path | Contents |
|---|---|
| `code/model_soft_delay.py` | Soft-Delay prediction backbone: action-conditioned graph generation and lag-aligned temporal aggregation |
| `code/model_ad_stgn.py` | Diagnostic encoder used by the root cause analysis module |
| `code/engine.py` | Training and evaluation loop for the prediction model |
| `code/utils.py` | Batch loading, standardisation, and masked error metrics |
| `code/tep_meta.py` | Process-unit definitions, fault ground truth, and reference routes |
| `code/run_monitor.py` | Monitoring chain: prediction residuals, control-chart alarm rule, and alarm-triggered diagnosis |
| `code/run_rca.py` | Root cause analysis: source-unit ranking and propagation-path reconstruction |
| `supplementary/` | Reference paths, process-influence edges, fault labels, and variable-to-unit mappings |

## Requirements

Python 3.10 or later, then

```bash
python -m pip install -r requirements.txt
```

CUDA is optional. Add `--device cpu` to the commands below on machines without
a GPU.

## Required input files

Observation data are not distributed with this repository. Create a data
directory (default `data/processed_tep`) and place the following files in it.

| File | Description |
|---|---|
| `train.npz`, `val.npz` | Normalisation statistics estimated on fault-free training and validation records |
| `A_static.npy` | Steady-state dependency graph, `40 x 40` |
| `target_lag_prior.npy` | Target-relevance lag prior |
| `rca_model.pth` | Weights of the diagnostic encoder |
| `monitor_run*.pth` | One or more prediction checkpoints; the checkpoint with the lowest validation residual is selected |
| `A_static_predictor.npy`, `tau_prior.npy` | Static graph and lag prior used by the prediction backbone |
| `<prefix>_f<fault>_e<event>.npz` | One file per evaluation event, holding the sensor windows, operating actions, target variable, and fault-onset index |

The event files are derived from the fault scenarios of the benchmark process.
The default prefix is `tep`, so an event of fault 1 is named
`tep_f1_e0.npz`.

## Usage

Root cause analysis at a fixed post-fault horizon:

```bash
python code/run_rca.py --data_dir data/processed_tep --device cpu \
    --trigger_mode fixed --fixed_delay 60 --out_dir results/rca
```

Monitoring with the control-chart alarm rule and alarm-triggered diagnosis:

```bash
python code/run_monitor.py --data_dir data/processed_tep --device cpu \
    --out_dir results/monitor
```

The diagnostic configuration is fixed in `code/run_rca.py`: three evidence
branches over the steady-state and event-dependent dependency structures, an
action-adaptive fusion weight, a bounded beam search, and a near-optimal path
selection rule. `code/tep_meta.py` holds the unit definitions and the
reference routes used to evaluate the reconstructed paths.

## Citation and licence

Please cite the accompanying manuscript and this repository; see
`CITATION.cff`. The author-owned code is released under the MIT licence
(`LICENSE`). The licence does not cover third-party benchmark datasets.
