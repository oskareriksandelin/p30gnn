import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class EquivariantGNN(MessagePassing):
    def __init__(self, node_feat_dim, hidden_dim=64, num_layers=3):
        super().__init__(aggr='add')  # sum aggregation

        self.num_layers = num_layers

        # MLP for edge messages (invariant scalars)
        self.edge_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(8, hidden_dim),  # 4 invariants + 4 node type scalars
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU()
            ) for _ in range(num_layers)
        ])

        # Node update MLP (optional, can just pass aggregated message to output)
        self.node_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU()
            ) for _ in range(num_layers)
        ])

        # Final linear layer to predict vector B-field
        self.final_lin = nn.Linear(hidden_dim, 3)  # output scalar per edge to multiply with r_ij

    def forward(self, data):
        # x: [num_nodes, node_feat_dim]  (moment + node_type)
        # edge_attr: [num_edges, 3]      (r_ij)
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for layer in range(self.num_layers):
            h = self.propagate(edge_index, x=x, edge_attr=edge_attr, mlp_edge=self.edge_mlps[layer])
            x = self.node_mlps[layer](h)
        return self.final_lin(x)

    def message(self, x_i, x_j, edge_attr, mlp_edge):
        """
        x_i, x_j: [num_edges, node_feat_dim]
        edge_attr: [num_edges, 4] -> [ux, uy, uz, |r_ij|]
        """
        # Spin vectors
        m_i = x_i[:, 2:5]
        m_j = x_j[:, 2:5]

        # Node type one-hot
        node_type_i = x_i[:, :2]
        node_type_j = x_j[:, :2]

        # Reconstruct actual r_ij vector
        u = edge_attr[:, :3]           # unit vector
        r_norm = edge_attr[:, 3:4]     # |r_ij|
        r_ij = u * r_norm              # [num_edges, 3]

        # Rotation-invariant scalars
        m_j_norm = torch.norm(m_j, dim=-1, keepdim=True)
        m_i_dot_m_j = (m_i * m_j).sum(dim=-1, keepdim=True)
        m_j_dot_r_ij = (m_j * r_ij).sum(dim=-1, keepdim=True)

        # Concatenate scalars + node types
        edge_features = torch.cat([
            m_j_norm,           # |m_j|
            m_i_dot_m_j,        # m_i · m_j
            m_j_dot_r_ij,       # m_j · r_ij
            r_norm,             # |r_ij|
            node_type_i,        # 2
            node_type_j         # 2
        ], dim=-1)  # total 8 features

        # Compute scalar message per hidden_dim
        msg_scalar = mlp_edge(edge_features)  # [num_edges, hidden_dim]

        # Multiply by r_ij to get vector message (SO(3) equivariant)
        msg_vector = msg_scalar  # broadcasting over last dim
        return msg_vector

    def update(self, aggr_out, x=None):
        # aggr_out is already a vector per node [num_nodes, 3]
        return aggr_out