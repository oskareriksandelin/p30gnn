import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class NaiveEdgeConv(MessagePassing):
    """GNN layer that uses all edge features for message passing, ignoring equivariance."""
    def __init__(self, in_features, hidden_dim, out_features, dropout=0.1):
        super().__init__(aggr='add')

        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),  # Add regularization
            nn.Linear(hidden_dim, out_features)
        )

        # Use ALL edge features: node_i + node_j + [dx,dy,dz,rij]
        self.edge_mlp = nn.Sequential(
            nn.Linear(13, hidden_dim),  # +4 for full edge
            nn.SiLU(),
            nn.Dropout(dropout),  # Add regularization
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )


    def forward(self, x, edge_index, edge_attr):
        # edge_attr should contain [dx, dy, dz, rij] as unit vectors + distance
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # use all edge_attr [dx, dy, dz, rij]

        # Spin vectors
        m_i = x_i[:, 2:5]
        m_j = x_j[:, 2:5]

        # Node type one-hot
        node_type_i = x_i[:, :2]
        node_type_j = x_j[:, :2]

        # Reconstruct actual r_ij vector
        u = edge_attr[:, :3]           # unit vector
        r_norm = edge_attr[:, 3:4]     # |r_ij|

        # Rotation-invariant scalars
        m_j_norm = torch.norm(m_j, dim=-1, keepdim=True)
        m_i_dot_m_j = (m_i * m_j).sum(dim=-1, keepdim=True)
        m_j_dot_r_ij = (m_j * u).sum(dim=-1, keepdim=True)

        # Concatenate scalars + node types
        edge_features = torch.cat([
            m_i,
            m_j,           # |m_j|
            m_i_dot_m_j,        # m_i · m_j
            m_j_dot_r_ij,       # m_j · r_ij
            r_norm,             # |r_ij|
            node_type_i,        # 2
            node_type_j         # 2
        ], dim=-1)  # total 13 features

        #edge_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_mlp(edge_features)

    def update(self, aggr_out, x):
        node_input = torch.cat([x, aggr_out], dim=-1)
        return self.node_mlp(node_input)
    
class NaiveFeGdBFieldModel(nn.Module):
    """
    GNN for predicting B-field from atomic moments using naive edge features (non-equivariant).

    Args:
        node_in_dim (int): Dimension of input node features.
        hidden_dim (int): Dimension of hidden layers.
        num_layers (int): Number of message passing layers.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(self, node_in_dim=5, hidden_dim=128, num_layers=4, dropout=0.1):
        super().__init__()

        # Initial embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.SiLU()
        )

        # Message passing layers
        self.conv_layers = nn.ModuleList([
            NaiveEdgeConv(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output head (per-atom B-field prediction)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)  # Bx, By, Bz
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Embed node features
        x = self.node_embedding(x)

        # Message passing
        for conv in self.conv_layers:
            x = x + conv(x, edge_index, edge_attr)  # Residual connection

        # Predict B-field for each atom
        b_field = self.output_mlp(x)

        return b_field