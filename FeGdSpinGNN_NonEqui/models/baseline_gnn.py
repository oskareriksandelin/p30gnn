import torch
from torch import nn
from torch_geometric.nn import MessagePassing

# Number of scalar features we build per edge:
# m_i (3) + m_j (3) + m_i·m_j (1) + m_j·û_ij (1) + |r_ij| (1) + type_i (2) + type_j (2) = 13
EDGE_FEATURE_DIM = 13


class NaiveEdgeConv(MessagePassing):
    """
    Simple message-passing layer that uses hand-crafted, non-equivariant
    edge features built from spins, node types and geometry.
    """
    def __init__(self, node_hidden_dim: int, edge_hidden_dim: int, dropout: float = 0.1):
        # "add" aggregation = sum of messages into each node
        super().__init__(aggr="add")

        # MLP that turns edge features into an edge message vector
        self.edge_mlp = nn.Sequential(
            nn.Linear(EDGE_FEATURE_DIM, edge_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.SiLU(),
        )

        # MLP that updates node states from previous state + aggregated messages
        self.node_mlp = nn.Sequential(
            nn.Linear(node_hidden_dim + edge_hidden_dim, node_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_hidden_dim, node_hidden_dim),
        )

    def forward(self,
                h: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                raw_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:       [num_nodes, node_hidden_dim] current node embeddings
            edge_index: [2, num_edges] sender/receiver indices
            edge_attr:  [num_edges, 4] (dx, dy, dz) unit vector + distance
            raw_x:  [num_nodes, 5] original node features
                   (2 one-hot type + 3 spin components)
        """
        return self.propagate(edge_index, x=h, edge_attr=edge_attr, raw_x=raw_x)

    def message(self,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                raw_x_i: torch.Tensor,
                raw_x_j: torch.Tensor) -> torch.Tensor:
        """
        Build edge features from *raw* physical features, not the hidden state.
        """
        # Spins (3-d) and node types (2-d one-hot) from raw features
        m_i = raw_x_i[:, 2:5]
        m_j = raw_x_j[:, 2:5]
        node_type_i = raw_x_i[:, :2]
        node_type_j = raw_x_j[:, :2]

        # Geometry: unit vector û_ij and distance |r_ij|
        u_ij = edge_attr[:, :3]      # shape [E, 3]
        r_norm = edge_attr[:, 3:4]   # shape [E, 1]

        # Simple rotation-invariant scalars
        m_i_dot_m_j = (m_i * m_j).sum(dim=-1, keepdim=True)
        m_j_dot_r = (m_j * u_ij).sum(dim=-1, keepdim=True)

        # Concatenate all scalars into a 13-d edge feature
        edge_features = torch.cat(
            [
                m_i,            # 3
                m_j,            # 3
                m_i_dot_m_j,    # 1
                m_j_dot_r,      # 1
                r_norm,         # 1
                node_type_i,    # 2
                node_type_j,    # 2
            ],
            dim=-1,
        )  # [E, 13]

        return self.edge_mlp(edge_features)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Combine old node embedding with aggregated messages.
        """
        node_input = torch.cat([x, aggr_out], dim=-1)
        return self.node_mlp(node_input)


class NaiveFeGdBFieldModel(nn.Module):
    """
    GNN for predicting B-field from atomic moments using naive, non-equivariant edge features.

    Args:
        node_in_dim:   dimension of input node features (default: 5).
        hidden_dim:    dimension of node embeddings.
        edge_hidden_dim: dimension of edge message embeddings.
        num_layers:    number of message passing layers.
        dropout:       dropout rate for regularization.
    """
    def __init__(self,
                 node_in_dim: int = 5,
                 hidden_dim: int = 128,
                 edge_hidden_dim: int = 128,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        # Initial embedding from raw features to hidden_dim
        self.node_embedding = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.SiLU(),
        )

        # Stack of message-passing layers
        self.conv_layers = nn.ModuleList(
            [
                NaiveEdgeConv(
                    node_hidden_dim=hidden_dim,
                    edge_hidden_dim=edge_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Per-atom B-field prediction head
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),  # Bx, By, Bz
        )

    def forward(self, data) -> torch.Tensor:
        """
        Args:
            data: PyG Data object with
                  x          [N, 5]   node features (2 type + 3 spin)
                  edge_index [2, E]   edges
                  edge_attr  [E, 4]   (dx, dy, dz, |r_ij|)
        Returns:
            b_field: [N, 3] predicted B-field per atom
        """
        raw_x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Embed node features
        h = self.node_embedding(raw_x)

        # Message passing with residual connections
        for conv in self.conv_layers:
            h = h + conv(h, edge_index, edge_attr, raw_x=raw_x)

        # Per-node B-field prediction
        b_field = self.output_mlp(h)
        return b_field
