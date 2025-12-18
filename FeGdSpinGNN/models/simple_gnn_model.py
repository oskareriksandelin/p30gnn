import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d

class SpinConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__(aggr='add')  # "Add" aggregation
        # MLP to process concatenated node features and edge attributes
        self.mlp = Sequential(
            Linear(in_channels + edge_dim, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )
        
    def forward(self, x, edge_index, edge_attr):
        # Compute messages and aggregate
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        # x_j has shape [num_edges, in_channels]
        # edge_attr has shape [num_edges, 2] if edge attributes are vectors in 2D
        tmp = torch.cat([x_j, edge_attr], dim=-1) # shape [num_edges, in_channels + edge_dim]
        return self.mlp(tmp)


class SimpleGNNModel(torch.nn.Module):
    """
    GNN model to predict magnetic B-field at each atom given atomic features and edge attributes.
    
    Args:
        input_dim (int): Dimension of input node features (e.g., atomic features + spin vectors).
        hidden_dim (int): Dimension of hidden layers.
        output_dim (int): Dimension of output features (e.g., 3 for B-field vector).
        num_layers (int): Number of message passing layers.
        edge_dim (int): Dimension of edge attributes (e.g., relative position vectors).
    """
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=3, num_layers=3, edge_dim=4):
        super().__init__()
        # Initialize the input projection layer 
        self.input_proj = Linear(input_dim, hidden_dim)
        
        # Inital message passing layer
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # Stack multiple SpinConv layers
        for _ in range(num_layers):
            self.convs.append(SpinConvLayer(hidden_dim, hidden_dim, edge_dim=edge_dim))
            self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Output MLP to predict B-field per node
        self.output_mlp = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Initial projection
        x = self.input_proj(x)  # [num_nodes, hidden_dim]
        x = F.relu(x)
        
        # Message passing layers
        for conv, bn in zip(self.convs, self.batch_norms):
            x_new = conv(x, edge_index, edge_attr)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x = x + x_new # Residual connection
            
        # Output
        out = self.output_mlp(x) # Predict B-field per node. [num_nodes, 3]
        return out
    
