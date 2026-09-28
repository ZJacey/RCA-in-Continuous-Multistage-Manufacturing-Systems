import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Module 1: action-driven graph generator (dynamic graph construction)
class ActionGraphGenerator(nn.Module):
    def __init__(self, action_dim, d, num_nodes, r=8):
        super(ActionGraphGenerator, self).__init__()
        self.num_nodes = num_nodes
        self.d = d
        self.r = r

        # Action Embedding (U -> E_u)
        self.emb_linear = nn.Linear(action_dim, d)

        # Low-Rank Graph Generation (W_src, W_tgt)
        self.W_s1 = nn.Linear(d, num_nodes * r)
        self.W_s2 = nn.Linear(d, num_nodes * r)

        # Action-gated cross-attention (query and key gating)
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_gQ = nn.Linear(d, d)
        self.W_gK = nn.Linear(d, d)

        # Learnable coefficient for dynamic/static graph fusion
        self.lam = nn.Parameter(torch.FloatTensor([0.5]))

    def forward(self, U, H_0, M_mask, A_static):
        """
        U: [Batch, T, action_dim]
        H_0: [Batch, T, num_nodes, d] initial node features
        M_mask: [num_nodes, num_nodes] admissibility mask
        A_static: [num_nodes, num_nodes] steady-state dependency graph
        """
        B, T, _ = U.shape

        # Step 1: Action Embedding
        E_u = F.relu(self.emb_linear(U))  # [B, T, d]

        # Step 2: Low-Rank Graph Generation
        W_src = self.W_s1(E_u).view(B, T, self.num_nodes, self.r)  # [B, T, N, r]
        W_tgt = self.W_s2(E_u).view(B, T, self.num_nodes, self.r)  # [B, T, N, r]
        # W_mod = W_src * W_tgt^T
        W_mod = torch.einsum('btnr, btmr -> btnm', W_src, W_tgt)  # [B, T, N, N]

        # Step 3: Action-Gated Cross-Attention
        g_Q = torch.sigmoid(self.W_gQ(E_u)).unsqueeze(2)  # [B, T, 1, d]
        g_K = torch.sigmoid(self.W_gK(E_u)).unsqueeze(2)  # [B, T, 1, d]

        Q_X = self.W_Q(H_0) * g_Q  # [B, T, N, d]
        K_X = self.W_K(H_0) * g_K  # [B, T, N, d]

        # Attn_u = Softmax(Q_X * K_X^T / sqrt(d))
        scores = torch.einsum('btnd, btmd -> btnm', Q_X, K_X) / math.sqrt(self.d)
        Attn_u = F.softmax(scores, dim=-1)  # [B, T, N, N]

        # Step 4: Physical Mask Fusion
        # Broadcast M_mask and A_static to [B, T, N, N]
        M_mask = M_mask.unsqueeze(0).unsqueeze(0)
        A_static = A_static.unsqueeze(0).unsqueeze(0)

        # Fuse the dynamic and static graphs, then apply the admissibility mask.
        A_total = ((W_mod * Attn_u) + self.lam * A_static) * M_mask

        return A_total

# Module 2: Soft-Delay spatio-temporal block
# Gated temporal convolution combined with lag-aligned spatial aggregation
class SoftDelaySTBlock(nn.Module):
    def __init__(self, d, num_nodes, tau_max, dilation):
        super(SoftDelaySTBlock, self).__init__()
        self.tau_max = tau_max

        # Factorised delay scoring to avoid an N*N*tau parameter tensor
        self.phi = nn.Parameter(torch.randn(num_nodes, tau_max))
        self.psi = nn.Parameter(torch.randn(num_nodes, tau_max))

        # Spatial projection
        self.spatial_proj = nn.Linear(d, d)

        # Gated temporal convolution over the sequence axis
        # Causal padding
        self.filter_conv = nn.Conv1d(in_channels=d, out_channels=d, kernel_size=2, dilation=dilation)
        self.gate_conv = nn.Conv1d(in_channels=d, out_channels=d, kernel_size=2, dilation=dilation)

        self.layer_norm = nn.LayerNorm(d)

    def forward(self, H, A_total, delta_prior):
        """
        H: [Batch, T, num_nodes, d] input features
        A_total: [Batch, T, num_nodes, num_nodes] dynamic graph
        delta_prior: [num_nodes, num_nodes, tau_max] engineering lag prior
        """
        B, T, N, D = H.shape

        # --- Step 5: Soft Delay Distribution ---
        # Reconstruct W_delay: [N, N, tau_max]
        phi_exp = self.phi.unsqueeze(1).expand(N, N, self.tau_max)
        psi_exp = self.psi.unsqueeze(0).expand(N, N, self.tau_max)
        W_delay_logits = phi_exp + psi_exp + delta_prior
        W_delay = F.softmax(W_delay_logits, dim=-1)  # [N, N, tau_max]

        # Step 6: sliding-window unfolding and tensor contraction
        # Pad the time axis so that tau_max historical steps are available.
        H_pad = F.pad(H.transpose(1, 3), (self.tau_max - 1, 0)).transpose(1, 3)  # [B, T+tau_max-1, N, D]

        # Unfold the historical window
        # H_pad.unfold yields [Batch, Time, Nodes, Dim, tau_max].
        H_window = H_pad.unfold(1, self.tau_max, 1)

        # Rearrange to the btjkc layout required by einsum: [Batch, Time, Nodes, tau_max, Dim].
        # The last two axes are swapped.
        H_window = H_window.transpose(-1, -2).contiguous()

        # Tensor contraction without Python loops.
        # Fuse the dynamic graph with the delay distribution.
        # A_delay[B, T, N_i, N_j, tau] combines A_total with W_delay.
        # The routing tensor keeps both spatial and temporal alignment.
        A_delay = A_total.unsqueeze(-1) * W_delay.unsqueeze(0).unsqueeze(0)

        # Contract it with the unfolded feature window.
        # The result is H_agg [B, T, N_i, D].
        H_agg = torch.einsum('btijk, btjkc -> btic', A_delay, H_window)

        # Spatial nonlinearity
        H_spa = F.relu(self.spatial_proj(H_agg))  # [B, T, N, D]

        # --- Step 7: Gated-TCN ---
        # Reshape for Conv1D: [B*N, D, T]
        H_tcn_in = H_spa.permute(0, 2, 3, 1).reshape(B * N, D, T)

        # Pad on the left to keep the sequence length T.
        pad_len = self.filter_conv.dilation[0]
        H_tcn_pad = F.pad(H_tcn_in, (pad_len, 0))

        filter_out = torch.tanh(self.filter_conv(H_tcn_pad))
        gate_out = torch.sigmoid(self.gate_conv(H_tcn_pad))
        Z = filter_out * gate_out  # [B*N, D, T]

        Z = Z.view(B, N, D, T).permute(0, 3, 1, 2)  # Restore [B, T, N, D]

        # Step 8: residual connection and layer normalisation
        H_out = self.layer_norm(Z + H)

        return H_out

# Module 3: AD-STGN network assembly
class AD_STGN(nn.Module):
    def __init__(self, node_features, action_dim, d, num_nodes, tau_max, num_layers=3, out_dim=1):
        super(AD_STGN, self).__init__()

        # Input projection
        self.input_proj = nn.Linear(node_features, d)

        # Graph generator
        self.graph_gen = ActionGraphGenerator(action_dim, d, num_nodes)

        # Stacked spatio-temporal blocks (dilation 1, 2, 4)
        self.st_blocks = nn.ModuleList([
            SoftDelaySTBlock(d, num_nodes, tau_max, dilation=2 ** i)
            for i in range(num_layers)
        ])

        # Prediction head (last time step)
        self.output_mlp = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, out_dim)
        )

    def forward(self, X, U, M_mask, A_static, delta_prior):
        """
        X: [B, T, N, C]
        """
        # 1. Initial node features
        H = self.input_proj(X)  # [B, T, N, d]

        # 2. Generate the dynamic graph sequence
        A_total = self.graph_gen(U, H, M_mask, A_static)  # [B, T, N, N]

        # 3. Pass through the spatio-temporal blocks
        for block in self.st_blocks:
            H = block(H, A_total, delta_prior)

        # 4. Predict from the last time step
        H_final = H[:, -1, :, :]  # [B, N, d]

        # Pool over nodes to obtain the global quality prediction.
        H_pool = H_final.mean(dim=1)  # [B, d] global quality prediction
        Y_hat = self.output_mlp(H_pool)  # [B, out_dim]

        return Y_hat, A_total  # A_total is returned for root cause analysis.
