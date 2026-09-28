# -*- coding: utf-8 -*-
"""
Diagnostic encoder used by the root cause analysis module.

Design points:
1) Lag-1 cross-sensor residual dynamics are used instead of a free soft-delay head,
   because fault transients concentrate on short lags.
2) A virtual target node represents the monitored quality variable. The variable is
   excluded from the graph inputs, so no current-target information enters the graph.
3) The first hop of a reconstructed path is the learned target-attention edge from a
   sensor node to the virtual target node.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8

def causal_shift(x):
    z = torch.zeros_like(x[:, :1])
    return torch.cat([z, x[:, :-1]], dim=1)

class ActionGraphGenerator(nn.Module):
    """Signed dynamic sensor graph A[child, parent], conditioned only on past state/action."""
    def __init__(self, action_dim, d, num_nodes, r=8):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d = int(d)
        self.r = int(r)

        self.emb_linear = nn.Linear(action_dim, d)
        self.W_s1 = nn.Linear(d, num_nodes * r)
        self.W_s2 = nn.Linear(d, num_nodes * r)

        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_gQ = nn.Linear(d, d)
        self.W_gK = nn.Linear(d, d)

        # Soft static prior strength. Missing Lasso edges are NOT hard-deleted.
        self.static_scale_raw = nn.Parameter(torch.tensor(-0.4))

    def forward(self, U_context, H_context, M_mask, A_static):
        B, T, _ = U_context.shape
        E_u = F.relu(self.emb_linear(U_context))

        W_src = self.W_s1(E_u).view(B, T, self.num_nodes, self.r)
        W_tgt = self.W_s2(E_u).view(B, T, self.num_nodes, self.r)
        W_mod = torch.einsum("btnr,btmr->btnm", W_src, W_tgt)

        gQ = torch.sigmoid(self.W_gQ(E_u)).unsqueeze(2)
        gK = torch.sigmoid(self.W_gK(E_u)).unsqueeze(2)
        Q = self.W_Q(H_context) * gQ
        K = self.W_K(H_context) * gK
        attn = F.softmax(
            torch.einsum("btnd,btmd->btnm", Q, K) / math.sqrt(self.d),
            dim=-1
        )

        A_dyn = torch.tanh(W_mod) * attn
        static_scale = F.softplus(self.static_scale_raw)
        A = A_dyn + static_scale * A_static.unsqueeze(0).unsqueeze(0)

        if M_mask is not None:
            A = A * M_mask.unsqueeze(0).unsqueeze(0)

        eye = torch.eye(
            self.num_nodes, device=A.device, dtype=A.dtype
        ).view(1, 1, self.num_nodes, self.num_nodes)
        A = A * (1.0 - eye)

        # Row normalization keeps each child's parent distribution comparable.
        denom = A.abs().sum(dim=-1, keepdim=True).clamp_min(EPS)
        return A / denom

class Lag1STBlock(nn.Module):
    """One-step causal graph propagation + gated temporal convolution."""
    def __init__(self, d, num_nodes, dilation):
        super().__init__()
        self.spatial_proj = nn.Linear(d, d)
        self.filter_conv = nn.Conv1d(d, d, kernel_size=2, dilation=dilation)
        self.gate_conv = nn.Conv1d(d, d, kernel_size=2, dilation=dilation)
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, H, A_total):
        B, T, N, D = H.shape

        # Physical lag-1 graph message: parent H_j(t-1) -> child i(t).
        H_prev = causal_shift(H)
        H_agg = torch.einsum("btij,btjd->btid", A_total, H_prev)
        H_spa = F.relu(self.spatial_proj(H_agg))

        x = H_spa.permute(0, 2, 3, 1).reshape(B * N, D, T)
        pad = self.filter_conv.dilation[0]
        x = F.pad(x, (pad, 0))
        z = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        z = z.view(B, N, D, T).permute(0, 3, 1, 2)
        return self.layer_norm(z + H)

class VirtualTargetHead(nn.Module):
    """Virtual target node: explicit sensor-to-target readout edges."""
    def __init__(self, d, out_dim=1):
        super().__init__()
        self.key = nn.Linear(d, d, bias=False)
        self.query = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.value = nn.Linear(d, d, bias=False)
        self.output_mlp = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, out_dim),
        )

    def forward(self, H_final):
        # H_final: [B,N,D]
        K = self.key(H_final)
        score = torch.einsum("bnd,d->bn", K, self.query) / math.sqrt(K.shape[-1])
        alpha = F.softmax(score, dim=-1)          # sensor-to-target attention
        V = self.value(H_final)
        target_h = torch.einsum("bn,bnd->bd", alpha, V)
        y_hat = self.output_mlp(target_h)
        return y_hat, alpha

class AD_STGN(nn.Module):
    VERSION = "AD-STGN"

    def __init__(self, node_features, action_dim, d, num_nodes,
                 num_layers=3, out_dim=1):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.action_dim = int(action_dim)
        self.d = int(d)

        self.input_proj = nn.Linear(node_features, d)
        self.graph_gen = ActionGraphGenerator(action_dim, d, num_nodes)

        # Nuisance dynamics are explicitly separated from cross-sensor RCA edges.
        self.self_ar_raw = nn.Parameter(torch.full((num_nodes,), 1.0))
        self.action_dyn = nn.Linear(action_dim, num_nodes, bias=True)
        nn.init.zeros_(self.action_dyn.weight)
        nn.init.zeros_(self.action_dyn.bias)
        self.cross_gain_raw = nn.Parameter(torch.full((num_nodes,), -1.3862944))

        self.st_blocks = nn.ModuleList([
            Lag1STBlock(d, num_nodes, dilation=2 ** i)
            for i in range(num_layers)
        ])

        self.target_head = VirtualTargetHead(d, out_dim=out_dim)

    def _reconstruct_lag1(self, X, U, A_total):
        x = X[..., 0]                       # [B,T,N]
        x_prev = causal_shift(x)
        u_prev = causal_shift(U)

        self_gain = torch.tanh(self.self_ar_raw).view(1, 1, -1)
        self_pred = self_gain * x_prev
        action_pred = self.action_dyn(u_prev)
        baseline = self_pred + action_pred

        cross_raw = torch.einsum("btij,btj->bti", A_total, x_prev)
        cross_gain = torch.sigmoid(self.cross_gain_raw).view(1, 1, -1)
        cross_pred = cross_gain * cross_raw

        return {
            "baseline": baseline,
            "cross_pred": cross_pred,
            "x_recon": baseline + cross_pred,
            "valid_start": 1,
        }

    def forward(self, X, U, M_mask, A_static, return_aux=False):
        H0 = self.input_proj(X)

        # Graph at t only sees t-1 context.
        H_context = causal_shift(H0)
        U_context = causal_shift(U)
        A_total = self.graph_gen(U_context, H_context, M_mask, A_static)

        H = H0
        for blk in self.st_blocks:
            H = blk(H, A_total)

        y_hat, target_attn = self.target_head(H[:, -1, :, :])

        if not return_aux:
            return y_hat, A_total, target_attn

        rec = self._reconstruct_lag1(X, U, A_total)
        support = (A_static.abs() > 0).to(A_total.dtype)
        outside_mass = (
            A_total.abs()
            * (1.0 - support).unsqueeze(0).unsqueeze(0)
        ).sum(dim=-1).mean()

        aux = {
            **rec,
            "outside_prior_mass": outside_mass,
            "target_attn": target_attn,
        }
        return y_hat, A_total, target_attn, aux
