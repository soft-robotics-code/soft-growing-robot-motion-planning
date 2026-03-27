"""
network_i/model.py  —  GAT–Transformer–MLP trajectory planning network (Network I).

Input : PyG graph with 36-dimensional node features
Output: (T, 2) predicted trajectory coordinates
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from configs import GNN_DIM, TRANS_DIM, NUM_TRANS_LAYERS, NODE_FEAT_DIM, SOFT_ROBOT_FLEXIBILITY


class SoftRobotTrajectoryPlanner(nn.Module):
    """GAT + Transformer + MLP planner for soft growing robots."""

    def __init__(self, in_dim=NODE_FEAT_DIM, gnn_dim=GNN_DIM,
                 trans_dim=TRANS_DIM, num_layers=NUM_TRANS_LAYERS):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, gnn_dim), nn.LayerNorm(gnn_dim),
            nn.ReLU(), nn.Dropout(0.1),
        )
        self.cond_enc = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, gnn_dim),
        )

        self.gnn1 = GATConv(gnn_dim,     gnn_dim, heads=2, concat=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(gnn_dim * 2)
        self.gnn2 = GATConv(gnn_dim * 2, gnn_dim, heads=2, concat=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(gnn_dim * 2)
        self.gnn3 = GATConv(gnn_dim * 2, gnn_dim, heads=1, concat=True, dropout=0.1)

        self.lstm = nn.LSTM(gnn_dim, gnn_dim, batch_first=True,
                             bidirectional=True, num_layers=3)

        self.proj = nn.Linear(gnn_dim * 2, trans_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=trans_dim, nhead=4,
                                        dim_feedforward=512, dropout=0.05,
                                        batch_first=True, activation="gelu"),
            num_layers=num_layers,
        )
        self.head = nn.Sequential(
            nn.Linear(trans_dim, trans_dim), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(trans_dim, trans_dim // 2), nn.ReLU(),
            nn.Linear(trans_dim // 2, 2),
        )
        self.flexibility = nn.Parameter(torch.tensor(float(SOFT_ROBOT_FLEXIBILITY)))

    def forward(self, data, start_goal_condition=None):
        """
        Parameters
        ----------
        data : torch_geometric.data.Data
            .x (N, in_dim), .edge_index (2, E), .traj_indices (T,)
        start_goal_condition : Tensor  shape (4,) = [sx, sz, gx, gz]

        Returns
        -------
        Tensor  (T, 2)
        """
        x = self.input_proj(data.x)
        if start_goal_condition is not None:
            cond  = self.cond_enc(start_goal_condition)
            scale = torch.sigmoid(cond) * torch.sigmoid(self.flexibility) + 0.5
            x     = x * scale.unsqueeze(0)

        x = torch.relu(self.gnn1(x, data.edge_index)); x = self.norm1(x)
        x = torch.relu(self.gnn2(x, data.edge_index)); x = self.norm2(x)
        x = torch.relu(self.gnn3(x, data.edge_index))

        traj = x[data.traj_indices]
        lstm_out, _ = self.lstm(traj.unsqueeze(0))
        enc = self.transformer(self.proj(lstm_out))
        return self.head(enc.squeeze(0))
