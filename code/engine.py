# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.optim as optim

import utils as util
from model_ad_stgn import AD_STGN

class Trainer:
    """Quality prediction + simple lag-1 residual dynamics."""
    def __init__(self, scaler_y, node_features, action_dim, d, num_nodes,
                 num_layers, out_dim, lrate, wdecay, device,
                 lambda_dyn=0.10, lambda_prior=0.02):
        self.model = AD_STGN(
            node_features=node_features,
            action_dim=action_dim,
            d=d,
            num_nodes=num_nodes,
            num_layers=num_layers,
            out_dim=out_dim,
        ).to(device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lrate, weight_decay=wdecay
        )
        self.loss_fn = util.masked_mae
        self.scaler_y = scaler_y
        self.lambda_dyn = float(lambda_dyn)
        self.lambda_prior = float(lambda_prior)
        self.clip = 5.0

    def _losses(self, input_x, real_y, pred_y, aux):
        y_loss = self.loss_fn(pred_y, real_y, null_val=np.nan)

        target = input_x[..., 0]
        s = int(aux["valid_start"])
        tgt = target[:, s:, :]
        base = aux["baseline"][:, s:, :]
        cross = aux["cross_pred"][:, s:, :]

        base_loss = self.loss_fn(base, tgt, null_val=np.nan)

        # Cross graph learns only the residual left by own persistence + actions.
        residual_target = (tgt - base).detach()
        cross_loss = self.loss_fn(cross, residual_target, null_val=np.nan)

        # Keep only the two independent dynamics losses.
        dyn_loss = 0.5 * (base_loss + cross_loss)
        prior_loss = aux["outside_prior_mass"]

        total = (
            y_loss
            + self.lambda_dyn * dyn_loss
            + self.lambda_prior * prior_loss
        )
        return total, y_loss, dyn_loss, prior_loss, base_loss, cross_loss

    def train(self, input_x, input_u, real_y, M_mask, A_static):
        self.model.train()
        self.optimizer.zero_grad()

        pred, A_total, target_attn, aux = self.model(
            input_x, input_u, M_mask, A_static, return_aux=True
        )
        vals = self._losses(input_x, real_y, pred, aux)

        vals[0].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        pred_real = self.scaler_y.inverse_transform(pred.detach())
        y_real = self.scaler_y.inverse_transform(real_y)
        rmse = util.masked_rmse(
            pred_real, y_real, null_val=np.nan
        ).item()

        return tuple(v.item() for v in vals) + (rmse,)

    def eval(self, input_x, input_u, real_y, M_mask, A_static):
        self.model.eval()
        with torch.no_grad():
            pred, A_total, target_attn, aux = self.model(
                input_x, input_u, M_mask, A_static, return_aux=True
            )
            vals = self._losses(input_x, real_y, pred, aux)

            pred_real = self.scaler_y.inverse_transform(pred)
            y_real = self.scaler_y.inverse_transform(real_y)
            rmse = util.masked_rmse(
                pred_real, y_real, null_val=np.nan
            ).item()

        return (
            tuple(v.item() for v in vals)
            + (rmse, pred_real, A_total, target_attn)
        )
