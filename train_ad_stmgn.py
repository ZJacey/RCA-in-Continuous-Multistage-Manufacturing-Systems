"""
Main Execution Script
Part of the AD-STMGN framework.
Executes the training loop, robust evaluation, and PST-CBS Root Cause Analysis engine.
"""
import os
import time
import argparse
import random
import warnings
import numpy as np
import torch
import util_ad_stmgn as util
from engine_ad_stmgn import Trainer

warnings.filterwarnings('ignore', category=RuntimeWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data_dir', type=str, default='./data/processed_tep/')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--seed', type=int, default=42)

# Model structure hyperparameters
parser.add_argument('--num_nodes', type=int, default=40)
parser.add_argument('--action_dim', type=int, default=11)
parser.add_argument('--node_features', type=int, default=1)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--tau_max', type=int, default=24)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--out_dim', type=int, default=1)
parser.add_argument('--seq_x', type=int, default=24)
args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data_and_priors(data_dir, batch_size):
    data = {}
    for cat in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(data_dir, f"{cat}.npz"))
        data[f'x_{cat}'] = cat_data['x']
        data[f'u_{cat}'] = cat_data['u']
        data[f'y_{cat}'] = cat_data['y']

    mean_x = data['x_train'].mean(axis=(0, 1), keepdims=True)
    std_x = data['x_train'].std(axis=(0, 1), keepdims=True)
    std_x[std_x == 0] = 1.0
    scaler_x = util.StandardScaler(mean=mean_x, std=std_x)

    mean_u = data['u_train'].mean(axis=(0, 1), keepdims=True)
    std_u = data['u_train'].std(axis=(0, 1), keepdims=True)
    std_u[std_u == 0] = 1.0
    scaler_u = util.StandardScaler(mean=mean_u, std=std_u)

    mean_y = data['y_train'].mean()
    std_y = data['y_train'].std()
    if std_y == 0: std_y = 1.0
    scaler_y = util.StandardScaler(mean=mean_y, std=std_y)

    for cat in ['train', 'val', 'test']:
        data[f'x_{cat}'] = scaler_x.transform(data[f'x_{cat}'])
        data[f'u_{cat}'] = scaler_u.transform(data[f'u_{cat}'])
        data[f'y_{cat}'] = scaler_y.transform(data[f'y_{cat}'])

    dataloader = {}
    for cat in ['train', 'val', 'test']:
        dataloader[f'{cat}_loader'] = util.DataLoader(
            data[f'x_{cat}'], data[f'u_{cat}'], data[f'y_{cat}'], batch_size
        )

    tau_prior_np = np.load(os.path.join(data_dir, "tau_prior.npy"))

    dyno_path = os.path.join(data_dir, "A_static.npy")
    if os.path.exists(dyno_path):
        A_static_np_full = np.load(dyno_path)
        A_static_np = A_static_np_full[:args.num_nodes, :args.num_nodes]
        M_mask_np = (A_static_np > 0).astype(np.float32) + np.eye(args.num_nodes)
        M_mask_np = np.clip(M_mask_np, 0, 1)
    else:
        A_static_np = np.zeros((args.num_nodes, args.num_nodes))
        M_mask_np = np.ones((args.num_nodes, args.num_nodes))

    return dataloader, scaler_y, tau_prior_np, M_mask_np, A_static_np, data


def build_delta_prior(tau_prior_np, tau_max, alpha=0.05):
    N = tau_prior_np.shape[0]
    delta = np.zeros((N, N, tau_max))
    for i in range(N):
        for j in range(N):
            delay = int(tau_prior_np[i, j])
            if delay >= tau_max: delay = tau_max - 1
            if delay > 0: delta[i, j, delay] = alpha
    return torch.FloatTensor(delta)


def run_rca_pst_cbs(A_total_seq, W_delay_probs, target_node, target_time, U_seq, beam_width=3, max_depth=10, eta=0.1):
    print(f"\n[PST-CBS Engine Started] Target Node: X{target_node}, Alarm Time: t={target_time}")
    beams = [{'path': [(target_node, target_time)], 'score': 0.0}]
    completed_paths = []
    N = A_total_seq.shape[1]
    tau_max = W_delay_probs.shape[2]

    for depth in range(1, max_depth + 1):
        all_candidates = []
        for beam in beams:
            curr_node, curr_time = beam['path'][-1]
            if curr_time <= 0:
                if beam not in completed_paths: completed_paths.append(beam)
                continue
            curr_graph = A_total_seq[curr_time]

            for parent_node in range(N):
                if parent_node == curr_node: continue
                for k in range(tau_max):
                    parent_time = curr_time - k
                    if parent_time < 0: continue
                    if (parent_node, parent_time) in beam['path']: continue

                    edge_weight = float(curr_graph[curr_node, parent_node])
                    edge_weight = max(edge_weight, 1e-3)

                    log_A = np.log(edge_weight)
                    log_W = np.log(float(W_delay_probs[curr_node, parent_node, k]) + 1e-8)

                    new_beam = {'path': beam['path'] + [(parent_node, parent_time)],
                                'score': beam['score'] + log_A + log_W}
                    all_candidates.append(new_beam)

        if not all_candidates: break
        all_candidates = sorted(all_candidates, key=lambda x: x['score'], reverse=True)
        beams = all_candidates[:beam_width]

    final_candidates = beams + [p for p in completed_paths if p not in beams]
    if not final_candidates: return

    for cand in final_candidates:
        D = len(cand['path']) - 1
        cand['norm_score'] = cand['score'] / D if D > 0 else cand['score']

    final_candidates = sorted(final_candidates, key=lambda x: x['norm_score'], reverse=True)[:beam_width]
    scores = np.array([c['norm_score'] for c in final_candidates])

    if np.all(np.isinf(scores)):
        probs = np.ones(len(scores)) / len(scores)
    else:
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)

    print("-" * 100)
    print("Ranked End-to-End Quality Degradation Physical Propagation Chains:")
    for idx, (cand, prob) in enumerate(zip(final_candidates, probs)):
        path = cand['path']
        v_root, t_root = path[-1]
        check_time = max(0, t_root)
        delta_U = np.abs(U_seq[check_time] - U_seq[check_time - 1]) if check_time > 0 else np.abs(U_seq[check_time])
        top_u_idx, max_delta_u = np.argmax(delta_U), delta_U[np.argmax(delta_U)]

        root_cause_str = f"[CI-Induced: Abnormal Action Intervention U{top_u_idx + 1} (Magnitude: {max_delta_u:.2f})]" if max_delta_u > eta else f"[Internal Fault: Endogenous Fault at Node X{v_root}]"

        path_str = ""
        for i in range(len(path) - 1, 0, -1):
            p_node, p_time = path[i]
            c_node, c_time = path[i - 1]
            path_str += f"X{p_node}(t={p_time}) -[Delay {c_time - p_time}]-> "
        path_str += f"X{path[0][0]}(t={path[0][1]})"

        print(f"Rank {idx + 1} (Confidence: {prob * 100:.1f}%) | Score: {cand['norm_score']:.2f}")
        print(f"   Root Cause: {root_cause_str}\n   Propagation Path: {path_str}\n")
    print("-" * 100)


def main():
    device = torch.device(args.device)

    dataloader, scaler_y, tau_prior_np, M_mask_np, A_static_np, data = load_data_and_priors(args.data_dir, args.batch_size)
    delta_prior_tensor = build_delta_prior(tau_prior_np, args.tau_max).to(device)
    M_mask_tensor = torch.FloatTensor(M_mask_np).to(device)
    A_static_tensor = torch.FloatTensor(A_static_np).to(device)

    x_val_scaled, u_val_scaled, y_val_scaled = data['x_val'], data['u_val'], data['y_val']
    x_test_scaled, u_test_scaled, y_test_scaled = data['x_test'], data['u_test'], data['y_test']
    y_test_raw_physical = np.load(os.path.join(args.data_dir, "test.npz"))['y'][:, -1, :]
    exact_fault_start_idx = np.argmax(np.load(os.path.join(args.data_dir, "test.npz"))['label'] > 0)

    buffer_size = 800
    if exact_fault_start_idx > 10000:
        cut_idx = exact_fault_start_idx - buffer_size
        x_test_scaled = x_test_scaled[cut_idx:]
        u_test_scaled = u_test_scaled[cut_idx:]
        y_test_scaled = y_test_scaled[cut_idx:]
        y_test_raw_physical = y_test_raw_physical[cut_idx:]
        exact_fault_start_idx = buffer_size
        print(f"\n[*] Long test sequence detected. Early noise cropped. Current steady-state buffer: {buffer_size} steps.")
    else:
        print(f"\n[*] Standard test sequence detected. Retaining full buffer: {exact_fault_start_idx} steps.")

    all_mae, all_rmse, all_mape, all_fdd, all_time = [], [], [], [], []

    for run in range(args.runs):
        print(f"\n{'=' * 30} Starting Run {run + 1}/{args.runs} {'=' * 30}")
        
        # Apply strict seed for current run to ensure perfect reproducibility
        set_seed(args.seed + run)

        engine = Trainer(
            scaler_y=scaler_y, node_features=args.node_features, action_dim=args.action_dim,
            d=args.hidden_dim, num_nodes=args.num_nodes, tau_max=args.tau_max,
            num_layers=args.num_layers, out_dim=args.out_dim, lrate=args.learning_rate,
            wdecay=args.weight_decay, device=device
        )

        best_val_loss = float('inf')
        model_save_path = os.path.join(args.data_dir, f"best_ad_stmgn_model_run{run}.pth")

        start_train_time = time.time()
        for epoch in range(args.epochs):
            train_loss = []
            dataloader['train_loader'].shuffle()

            for x, u, y in dataloader['train_loader'].get_iterator():
                trainx, trainu, trainy = torch.FloatTensor(x).to(device), torch.FloatTensor(u).to(
                    device), torch.FloatTensor(y).to(device)[:, -1, :]
                metrics = engine.train(trainx, trainu, trainy, M_mask_tensor, A_static_tensor, delta_prior_tensor)
                train_loss.append(metrics[0])

            val_loss = []
            for x, u, y in dataloader['val_loader'].get_iterator():
                valx, valu, valy = torch.FloatTensor(x).to(device), torch.FloatTensor(u).to(device), torch.FloatTensor(
                    y)[:, -1, :].to(device)
                metrics = engine.eval(valx, valu, valy, M_mask_tensor, A_static_tensor, delta_prior_tensor)
                val_loss.append(metrics[0])

            mval_loss = np.mean(val_loss)

            if mval_loss < best_val_loss:
                best_val_loss = mval_loss
                torch.save(engine.model.state_dict(), model_save_path)

        end_train_time = time.time()
        run_time = end_train_time - start_train_time
        all_time.append(run_time)
        print(f"Training finished! Time taken: {run_time:.2f}s\n")

        engine.model.load_state_dict(torch.load(model_save_path))

        val_losses = []
        for x, u, y in zip(x_val_scaled, u_val_scaled, y_val_scaled):
            valx, valu, valy = torch.FloatTensor(x).unsqueeze(0).to(device), torch.FloatTensor(u).unsqueeze(0).to(
                device), torch.FloatTensor(y).unsqueeze(0)[:, -1, :].to(device)
            metrics = engine.eval(valx, valu, valy, M_mask_tensor, A_static_tensor, delta_prior_tensor)
            val_losses.append(metrics[0])

        mu_val = np.mean(val_losses)
        sigma_val = np.std(val_losses)
        sigma_coef = 3.0
        patience = 5
        UCL_threshold = mu_val + sigma_coef * sigma_val

        test_loss_scaled, real_absolute_errors = [], []
        alarm_triggered = False
        fault_A_total, fault_U = None, None
        delay = np.nan

        with torch.no_grad():
            for iter in range(len(x_test_scaled)):
                testx, testu, testy = torch.FloatTensor(x_test_scaled[iter]).unsqueeze(0).to(device), torch.FloatTensor(
                    u_test_scaled[iter]).unsqueeze(0).to(device), torch.FloatTensor(y_test_scaled[iter]).unsqueeze(0)[:, -1, :].to(device)
                metrics = engine.eval(testx, testu, testy, M_mask_tensor, A_static_tensor, delta_prior_tensor)

                current_loss_scaled = metrics[0]
                test_loss_scaled.append(current_loss_scaled)
                real_absolute_errors.append(current_loss_scaled * float(scaler_y.std))

                if current_loss_scaled > UCL_threshold and not alarm_triggered:
                    if iter > (patience - 1) and np.mean(test_loss_scaled[-patience:]) > UCL_threshold:
                        delay = iter - exact_fault_start_idx
                        if delay < 0:
                            print(f"[False Alarm] Triggered {abs(delay)} steps early.")
                        else:
                            print(f"[Success] Fault detected. Fault Detection Delay (FDD): {delay} sampling windows.")
                            fault_A_total, fault_U = metrics[3], testu
                        alarm_triggered = True
                        break

        all_fdd.append(delay)

        collected_len = len(real_absolute_errors)
        valid_idx = min(collected_len, exact_fault_start_idx)

        if valid_idx > 0:
            steady_state_errors = np.array(real_absolute_errors[:valid_idx]).reshape(-1)
            steady_state_trues = y_test_raw_physical[:valid_idx].reshape(-1)
            epsilon = 1e-2
            safe_denominator = np.maximum(np.abs(steady_state_trues), epsilon)

            final_mae = np.mean(steady_state_errors)
            final_rmse = np.sqrt(np.mean(steady_state_errors ** 2))
            final_mape = np.mean(steady_state_errors / safe_denominator) * 100

            all_mae.append(final_mae)
            all_rmse.append(final_rmse)
            all_mape.append(final_mape)
            print(f"Run MAE: {final_mae:.4f} | RMSE: {final_rmse:.4f} | MAPE: {final_mape:.4f}%")
        else:
            print("\n[Warning] No valid steady-state interval available for error calculation.")
            all_mae.append(np.nan)
            all_rmse.append(np.nan)
            all_mape.append(np.nan)

        if alarm_triggered and fault_A_total is not None and delay >= 0:
            with torch.no_grad():
                W_delay_logits = engine.model.st_blocks[0].phi.unsqueeze(1) + engine.model.st_blocks[0].psi.unsqueeze(
                    0) + delta_prior_tensor
                W_delay_probs = torch.softmax(W_delay_logits, dim=-1)
            run_rca_pst_cbs(A_total_seq=fault_A_total[0].cpu().numpy(), W_delay_probs=W_delay_probs.cpu().numpy(),
                            target_node=args.num_nodes - 1, target_time=fault_A_total[0].shape[0] - 1,
                            U_seq=fault_U[0].cpu().numpy(), beam_width=3, max_depth=10, eta=0.10)

    print("\n" + "=" * 60)
    print(f"Summary of {args.runs} Independent Runs (Mean +/- Std):")
    print(f"Training Time : {np.nanmean(all_time):.2f} +/- {np.nanstd(all_time):.2f} s")
    print(f"Test MAE      : {np.nanmean(all_mae):.4f} +/- {np.nanstd(all_mae):.4f}")
    print(f"Test RMSE     : {np.nanmean(all_rmse):.4f} +/- {np.nanstd(all_rmse):.4f}")
    print(f"Test MAPE     : {np.nanmean(all_mape):.4f}% +/- {np.nanstd(all_mape):.4f}%")
    print(f"FDD           : {np.nanmean(all_fdd):.2f} +/- {np.nanstd(all_fdd):.2f} steps")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()