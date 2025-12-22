"""
SIMPLIFIED Equivariant GNN for B-field prediction in Fe-Gd systems.

Key idea: 
1. Extract ONLY scalar features from spins: |m_i|, |m_j|, m_i·m_j, m_i·r̂, m_j·r̂
2. Process these scalars through MLPs
3. Output scalar weights per atom
4. Construct B-field as: B_i = Σ_j weight_j * spin_j

This ensures equivariance: if you rotate all spins → B-field rotates the same way.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class SimpleInvariantLayer(MessagePassing):
    """
    One message passing layer using ONLY scalar invariants.
    
    Input per edge:
    - Node features: [node_type (2), spin (3)]
    - Edge features: [unit_vector (3), distance (1)]
    
    What we compute:
    - Scalars: |m_i|, |m_j|, m_i·m_j, m_i·r̂, m_j·r̂, distance, node_types
    - Total: 6 spin scalars + 1 distance + 2+2 node types = 11 features
    """
    def __init__(self, hidden_dim):
        super().__init__(aggr='add')  # Sum messages from neighbors
        
        # MLP to process scalar edge features
        self.edge_mlp = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # MLP to update node features
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # [node_feat, aggregated_messages]
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, h, edge_index, x_original, edge_attr):
        """
        Args:
            h: hidden features [N, hidden_dim]
            edge_index: connectivity [2, E]
            x_original: original features [N, 5] = [node_type (2), spin (3)]
            edge_attr: edge features [E, 4] = [unit_vec (3), distance (1)]
        """
        return self.propagate(edge_index, h=h, x_original=x_original, edge_attr=edge_attr)
    
    def message(self, x_original_i, x_original_j, edge_attr):
        """Compute messages from scalars only."""
        # Extract components
        node_type_i = x_original_i[:, :2]  # [E, 2]
        node_type_j = x_original_j[:, :2]  # [E, 2]
        m_i = x_original_i[:, 2:5]         # [E, 3] spin vectors
        m_j = x_original_j[:, 2:5]         # [E, 3]
        
        unit_vec = edge_attr[:, :3]        # [E, 3] direction
        distance = edge_attr[:, 3:4]       # [E, 1]
        
        # Compute scalar invariants from spins
        m_i_norm = m_i.norm(dim=-1, keepdim=True)           # |m_i|
        m_j_norm = m_j.norm(dim=-1, keepdim=True)           # |m_j|
        m_i_dot_m_j = (m_i * m_j).sum(dim=-1, keepdim=True) # m_i · m_j
        m_i_dot_r = (m_i * unit_vec).sum(dim=-1, keepdim=True)  # m_i · r̂
        m_j_dot_r = (m_j * unit_vec).sum(dim=-1, keepdim=True)  # m_j · r̂
        
        # Concatenate ALL scalars (no vectors!)
        scalars = torch.cat([
            m_i_norm, m_j_norm, m_i_dot_m_j, m_i_dot_r, m_j_dot_r,  # 5 spin scalars
            distance,                                                 # 1 distance
            node_type_i, node_type_j                                 # 2+2 types
        ], dim=-1)  # [E, 11]
        
        return self.edge_mlp(scalars)
    
    def update(self, aggr_out, h):
        """Update node features with aggregated messages."""
        return self.node_mlp(torch.cat([h, aggr_out], dim=-1))


class InvariantFeGdBFieldModel(nn.Module):
    """
    Simplified equivariant GNN for B-field prediction.
    
    Architecture:
    1. Embed scalar features (node_type + |spin|)
    2. Message passing with scalar invariants
    3. Output scalar weight per atom
    4. Aggregate: B_i = Σ_j weight_j * spin_j
    
    Input format:
        data.x: [N, 5] = [node_type (2), spin_x, spin_y, spin_z]
        data.edge_index: [2, E]
        data.edge_attr: [E, 4] = [dx, dy, dz, distance]
    
    Output:
        B-field: [N, 3]
    """
    def __init__(self, hidden_dim=128, num_layers=3):
        super().__init__()
        
        # Initial embedding of scalar features
        self.input_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),  # node_type (2) + |spin| (1) = 3 scalars
            nn.SiLU()
        )
        
        # Message passing layers
        self.layers = nn.ModuleList([
            SimpleInvariantLayer(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Output: scalar weight per atom
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)  # Just one scalar weight
        )
    
    def forward(self, data):
        """
        Args:
            data: PyG Data with x, edge_index, edge_attr
        
        Returns:
            b_field: [N, 3] predicted B-field
        """
        x = data.x              # [N, 5]
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        # Extract components
        node_type = x[:, :2]    # [N, 2] one-hot
        spins = x[:, 2:5]       # [N, 3] spin vectors
        
        # Normalize spins to unit vectors (important!)
        spins = torch.nn.functional.normalize(spins, dim=1)
        
        # Create scalar features for embedding
        spin_norm = spins.norm(dim=-1, keepdim=True)  # Should be ~1.0 after normalization
        scalar_features = torch.cat([node_type, spin_norm], dim=-1)  # [N, 3]
        
        # Initial embedding
        h = self.input_mlp(scalar_features)  # [N, hidden_dim]
        
        # Reconstruct x_original for computing invariants
        x_original = torch.cat([node_type, spins], dim=-1)  # [N, 5]
        
        # Message passing with residual connections
        for layer in self.layers:
            h = h + layer(h, edge_index, x_original, edge_attr)
        
        # Compute scalar weight per atom
        weights = self.output_mlp(h)  # [N, 1]
        
        # Construct B-field: aggregate weighted neighbor spins
        # B_i = Σ_j weight_j * spin_j
        src, dst = edge_index  # src = neighbor j, dst = atom i
        
        neighbor_spins = spins[src]       # [E, 3] spins of neighbors
        neighbor_weights = weights[src]   # [E, 1] weights of neighbors
        
        # Weighted neighbor contributions
        weighted_contributions = neighbor_weights * neighbor_spins  # [E, 3]
        
        # Aggregate to each atom
        b_field = torch.zeros(spins.size(0), 3, device=spins.device, dtype=spins.dtype)
        b_field.scatter_add_(0, dst.unsqueeze(1).expand_as(weighted_contributions), weighted_contributions)
        
        return b_field


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    from torch_geometric.data import Data
    
    print("=" * 60)
    print("InvariantFeGdBFieldModel - Test")
    print("=" * 60)
    
    # Example data (your format)
    num_atoms = 800
    node_type = torch.eye(2)[torch.randint(0, 2, (num_atoms,))]  # Fe or Gd
    spins = torch.randn(num_atoms, 3)
    x = torch.cat([node_type, spins], dim=-1)  # [800, 5]
    
    # Random edges
    edge_index = torch.randint(0, num_atoms, (2, 9168))
    edge_attr = torch.randn(9168, 4)
    edge_attr[:, :3] = torch.nn.functional.normalize(edge_attr[:, :3], dim=1)  # unit vectors
    edge_attr[:, 3] = torch.rand(9168) * 0.2 + 0.1  # distances 0.1-0.3
    
    y = torch.randn(num_atoms, 3)  # Target B-field
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    # Initialize model
    model = InvariantFeGdBFieldModel(hidden_dim=128, num_layers=3)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        b_pred = model(data)
    
    print(f"\nInput shape: {data.x.shape}")
    print(f"Output shape: {b_pred.shape}")
    print(f"B-field range: [{b_pred.min():.4f}, {b_pred.max():.4f}]")
    
    # Test equivariance
    print("\n" + "=" * 60)
    print("Testing Equivariance")
    print("=" * 60)
    
    with torch.no_grad():
        b_original = model(data)
        
        # Random rotation
        R = torch.randn(3, 3)
        R, _ = torch.linalg.qr(R)
        if torch.det(R) < 0:
            R[:, 0] *= -1
        
        # Rotate spins and edge vectors
        node_type = data.x[:, :2]
        spins = data.x[:, 2:5]
        spins_rotated = (R @ spins.T).T
        x_rotated = torch.cat([node_type, spins_rotated], dim=-1)
        
        edge_attr_rotated = data.edge_attr.clone()
        edge_attr_rotated[:, :3] = (R @ data.edge_attr[:, :3].T).T
        
        data_rotated = Data(x=x_rotated, edge_index=data.edge_index, edge_attr=edge_attr_rotated, y=data.y)
        b_rotated = model(data_rotated)
        
        b_expected = (R @ b_original.T).T
        error = torch.norm(b_expected - b_rotated) / (torch.norm(b_rotated) + 1e-8)
        
        print(f"Equivariance error: {error.item():.6e}")
        if error < 1e-5:
            print("✓ Model is EQUIVARIANT!")
        else:
            print("✗ Model is NOT equivariant")