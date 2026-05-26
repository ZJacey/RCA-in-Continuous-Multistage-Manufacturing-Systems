"""
Data Preprocessing Script
Part of the AD-STMGN framework.
Handles the conversion of raw CSV files into processed tensor windows for training and testing.
"""
import os
import numpy as np
import pandas as pd

def process_tep_files(data_dir="./data/TEP/", 
                      output_dir="./data/processed_tep/"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading TEP dataset...")
    df_train = pd.read_csv(os.path.join(data_dir, "TEP_FaultFree_Training.csv"))
    df_test_free = pd.read_csv(os.path.join(data_dir, "TEP_FaultFree_Testing.csv"))
    df_test_faulty = pd.read_csv(os.path.join(data_dir, "TEP_Faulty_Testing.csv"))


    print("Constructing hybrid testing sequence...")
    run_normal = df_test_free[df_test_free['simulationRun'] == 1].copy()
    run_faulty = df_test_faulty[df_test_faulty['simulationRun'] == 1].copy()

    run_faulty['simulationRun'] = 999
    run_normal['simulationRun'] = 999

    df_test_hybrid = pd.concat([run_normal, run_faulty], ignore_index=True)

    def extract_tensors(df):
        target_var = 35
        y_col = [f'xmeas_{target_var}']
        
        x_cols = [f'xmeas_{i}' for i in range(1, 42) if i != target_var]
        u_cols = [f'xmv_{i}' for i in range(1, 12)]
        
        labels = df['faultNumber'].values if 'faultNumber' in df.columns else np.zeros(len(df))

        return df[x_cols].values, df[u_cols].values, df[y_col].values, labels

    X_tr_full, U_tr_full, Y_tr_full, _ = extract_tensors(df_train)
    X_te_full, U_te_full, Y_te_full, L_te_full = extract_tensors(df_test_hybrid)

    X_tr_full = np.expand_dims(X_tr_full, axis=-1)
    X_te_full = np.expand_dims(X_te_full, axis=-1)

    seq_x, seq_y, tau_max = 24, 1, 15

    def slide_window(df_raw, X, U, Y, L=None):
        x_list, u_list, y_list, l_list = [], [], [], []
        
        for run_id in df_raw['simulationRun'].unique():
            idx = df_raw.index[df_raw['simulationRun'] == run_id].tolist()
            if len(idx) < seq_x + seq_y + tau_max:
                continue

            X_run = X[idx[0]:idx[-1] + 1]
            U_run = U[idx[0]:idx[-1] + 1]
            Y_run = Y[idx[0]:idx[-1] + 1]
            if L is not None:
                L_run = L[idx[0]:idx[-1] + 1]

            for i in range(tau_max, len(X_run) - seq_x - seq_y + 1):
                x_list.append(X_run[i: i + seq_x])
                u_list.append(U_run[i: i + seq_x])
                y_list.append(Y_run[i + seq_x: i + seq_x + seq_y])
                if L is not None:
                    l_list.append(L_run[i + seq_x])

        if L is not None:
            return np.array(x_list), np.array(u_list), np.array(y_list), np.array(l_list)
        return np.array(x_list), np.array(u_list), np.array(y_list)

    print("Processing sliding windows...")
    X_tr_windows, U_tr_windows, Y_tr_windows = slide_window(df_train, X_tr_full, U_tr_full, Y_tr_full)
    X_test_windows, U_test_windows, Y_test_windows, L_test_windows = slide_window(df_test_hybrid, X_te_full, U_te_full, Y_te_full, L_te_full)

    num_train = int(len(X_tr_windows) * 0.8)

    print("Saving processed tensors...")
    np.savez_compressed(os.path.join(output_dir, "train.npz"),
                        x=X_tr_windows[:num_train], u=U_tr_windows[:num_train], y=Y_tr_windows[:num_train])
    np.savez_compressed(os.path.join(output_dir, "val.npz"),
                        x=X_tr_windows[num_train:], u=U_tr_windows[num_train:], y=Y_tr_windows[num_train:])
    np.savez_compressed(os.path.join(output_dir, "test.npz"),
                        x=X_test_windows, u=U_test_windows, y=Y_test_windows, label=L_test_windows)

    tau_prior = np.ones((40, 40)) * 2
    np.save(os.path.join(output_dir, "tau_prior.npy"), tau_prior)

    print("TEP dataset processing completed.")
    print(f"Train samples: {num_train} | Validation samples: {len(X_tr_windows) - num_train}")
    print(f"Test samples (Hybrid): {len(X_test_windows)}")

if __name__ == "__main__":
    process_tep_files()