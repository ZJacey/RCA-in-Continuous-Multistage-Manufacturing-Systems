"""
Trainer Engine
Part of the AD-STMGN framework.
Handles model initialization, optimization, training steps, and evaluation metrics.
"""
import torch
import torch.optim as optim
import numpy as np
import util_ad_stmgn as util
from model_ad_stmgn import AD_STMGN

class Trainer:
    def __init__(self, scaler_y, node_features, action_dim, d, num_nodes, tau_max, num_layers, out_dim, lrate, wdecay,
                 device):
        # Initialize the AD-STMGN model
        self.model = AD_STMGN(
            node_features=node_features,
            action_dim=action_dim,
            d=d,
            num_nodes=num_nodes,
            tau_max=tau_max,
            num_layers=num_layers,
            out_dim=out_dim
        )
        self.model.to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss_fn = util.masked_mae
        self.scaler_y = scaler_y
        self.clip = 5.0
        self.device = device

    def train(self, input_x, input_u, real_val_y, M_mask, A_static, delta_prior):
        self.model.train()
        self.optimizer.zero_grad()

        predict_scaled, _ = self.model(input_x, input_u, M_mask, A_static, delta_prior)

        # Compute loss in normalized space (using np.nan for null values)
        loss = self.loss_fn(predict_scaled, real_val_y, null_val=np.nan)

        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        # Inverse transform to monitor physical RMSE
        predict_real = self.scaler_y.inverse_transform(predict_scaled.detach())
        real_val_real = self.scaler_y.inverse_transform(real_val_y)
        rmse = util.masked_rmse(predict_real, real_val_real, null_val=np.nan).item()

        return loss.item(), rmse

    def eval(self, input_x, input_u, real_val_y, M_mask, A_static, delta_prior):
        self.model.eval()
        with torch.no_grad():
            predict_scaled, A_total_learned = self.model(input_x, input_u, M_mask, A_static, delta_prior)

            loss = self.loss_fn(predict_scaled, real_val_y, null_val=np.nan)

            predict_real = self.scaler_y.inverse_transform(predict_scaled)
            real_val_real = self.scaler_y.inverse_transform(real_val_y)
            rmse = util.masked_rmse(predict_real, real_val_real, null_val=np.nan).item()

        return loss.item(), rmse, predict_real, A_total_learned