"""
Model Architecture Definition
Part of the AD-STMGN framework.
Contains the Action-Driven Graph Generator, Soft-Delay ST-Block, and main AD_STMGN network.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Module 1: Action-Driven Graph Generator
class ActionGraphGenerator(nn.Module):
    def __init__(self, action_dim, d, num_nodes, r=8):
        super(ActionGraphGenerator, self).__init__()
        self.num_nodes = num_nodes
        self.d = d
        self.r = r

        self.emb_linear = nn.Linear(action_dim, d)

        # Weight Projection Layer
        self.W_s1 = nn.Linear(d, num_nodes * r)
        self.W_s2 = nn.Linear(d, num_nodes * r)

        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_gQ = nn.Linear(d, d)
        self.W_gK = nn.Linear(d, d)

        self.lam = nn.Parameter(torch.FloatTensor([0.5]))

    def forward(self, U, H_0, M_mask, A_static):
        B, T, _ = U.shape

        E_u = F.relu(self.emb_linear(U))

        W_src = self.W_s1(E_u).view(B, T, self.num_nodes, self.r)
        W_tgt = self.W_s2(E_u).view(B, T, self.num_nodes, self.r)
        
        W_mod = torch.einsum('btnr, btmr -> btnm', W_src, W_tgt)

        g_Q = torch.sigmoid(self.W_gQ(E_u)).unsqueeze(2)
        g_K = torch.sigmoid(self.W_gK(E_u)).unsqueeze(2)

        Q_X = self.W_Q(H_0) * g_Q
        K_X = self.W_K(H_0) * g_K

        scores = torch.einsum('btnd, btmd -> btnm', Q_X, K_X) / math.sqrt(self.d)
        Attn_u = F.softmax(scores, dim=-1)

        M_mask = M_mask.unsqueeze(0).unsqueeze(0)
        A_static = A_static.unsqueeze(0).unsqueeze(0)

        A_total = ((W_mod * Attn_u) + self.lam * A_static) * M_mask

        return A_total


# Module 2: Soft-Delay ST-Block
class SoftDelaySTBlock(nn.Module):
    def __init__(self, d, num_nodes, tau_max, dilation):
        super(SoftDelaySTBlock, self).__init__()
        self.tau_max = tau_max

        self.phi = nn.Parameter(torch.randn(num_nodes, tau_max))
        self.psi = nn.Parameter(torch.randn(num_nodes, tau_max))

        self.spatial_proj = nn.Linear(d, d)

        self.filter_conv = nn.Conv1d(in_channels=d, out_channels=d, kernel_size=2, dilation=dilation)
        self.gate_conv = nn.Conv1d(in_channels=d, out_channels=d, kernel_size=2, dilation=dilation)

        self.layer_norm = nn.LayerNorm(d)

    def forward(self, H, A_total, delta_prior):
        B, T, N, D = H.shape

        phi_exp = self.phi.unsqueeze(1).expand(N, N, self.tau_max)
        psi_exp = self.psi.unsqueeze(0).expand(N, N, self.tau_max)
        W_delay_logits = phi_exp + psi_exp + delta_prior
        W_delay = F.softmax(W_delay_logits, dim=-1)

        H_pad = F.pad(H.transpose(1, 3), (self.tau_max - 1, 0)).transpose(1, 3)

        H_window = H_pad.unfold(1, self.tau_max, 1)

        H_window = H_window.transpose(-1, -2).contiguous()

        A_delay = A_total.unsqueeze(-1) * W_delay.unsqueeze(0).unsqueeze(0)

        H_agg = torch.einsum('btijk, btjkc -> btic', A_delay, H_window)

        H_spa = F.relu(self.spatial_proj(H_agg))

        H_tcn_in = H_spa.permute(0, 2, 3, 1).reshape(B * N, D, T)

        pad_len = self.filter_conv.dilation[0]
        H_tcn_pad = F.pad(H_tcn_in, (pad_len, 0))

        filter_out = torch.tanh(self.filter_conv(H_tcn_pad))
        gate_out = torch.sigmoid(self.gate_conv(H_tcn_pad))
        Z = filter_out * gate_out

        Z = Z.view(B, N, D, T).permute(0, 3, 1, 2)

        H_out = self.layer_norm(Z + H)

        return H_out


# Module 3: AD-STMGN Network
class AD_STMGN(nn.Module):
    def __init__(self, node_features, action_dim, d, num_nodes, tau_max, num_layers=3, out_dim=1):
        super(AD_STMGN, self).__init__()

        self.input_proj = nn.Linear(node_features, d)

        self.graph_gen = ActionGraphGenerator(action_dim, d, num_nodes)

        self.st_blocks = nn.ModuleList([
            SoftDelaySTBlock(d, num_nodes, tau_max, dilation=2 ** i)
            for i in range(num_layers)
        ])

        self.output_mlp = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, out_dim)
        )

    def forward(self, X, U, M_mask, A_static, delta_prior):
        H = self.input_proj(X)

        A_total = self.graph_gen(U, H, M_mask, A_static)

        for block in self.st_blocks:
            H = block(H, A_total, delta_prior)

        H_final = H[:, -1, :, :]

        H_pool = H_final.mean(dim=1)
        Y_hat = self.output_mlp(H_pool)

        return Y_hat, A_total