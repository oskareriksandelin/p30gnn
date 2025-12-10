import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from e3nn import o3
from e3nn.nn import Gate
from e3nn.o3 import Linear, TensorProduct, FullyConnectedTensorProduct
from typing import Optional


class E3EquivariantMPNN(MessagePassing):
    """
    Equivariant Message Passing Layer for magnetic field prediction.
    Processes scalar features (node types) and vector features (spin moments).
    """
    def __init__(
        self,
        irreps_node_input,
        irreps_node_hidden,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons=[64, 64],
        num_neighbors=None,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_hidden = o3.Irreps(irreps_node_hidden)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)

        # Tensor product for combining node features with edge attributes
        self.tp = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_node_input,
            irreps_in2=self.irreps_edge_attr,
            irreps_out=self.irreps_node_hidden,
            shared_weights=False,
            internal_weights=False,
        )

        # MLP for generating tensor product weights
        self.fc = nn.Sequential(
            nn.Linear(self.irreps_edge_attr.dim, fc_neurons[0]), # Changed from .num_irreps to .dim
            nn.SiLU(),
            nn.Linear(fc_neurons[0], fc_neurons[1]),
            nn.SiLU(),
            nn.Linear(fc_neurons[1], self.tp.weight_numel)
        )

        # Self-interaction
        self.self_interaction = Linear(
            irreps_in=self.irreps_node_input,
            irreps_out=self.irreps_node_hidden
        )

        # Output projection
        self.linear_output = Linear(
            irreps_in=self.irreps_node_hidden,
            irreps_out=self.irreps_node_output
        )

        # Normalization factor
        self.num_neighbors = num_neighbors

    def forward(self, x, edge_index, edge_attr, edge_scalars=None):
        # x: node features [N, irreps_node_input]
        # edge_index: [2, E]
        # edge_attr: edge spherical harmonics [E, irreps_edge_attr]
        # edge_scalars: edge scalar features [E, num_scalars] for weight generation

        # Self-interaction
        x_self = self.self_interaction(x)

        # Message passing
        x_msg = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_scalars=edge_scalars)

        # Combine and output
        x_out = x_self + x_msg

        if self.num_neighbors is not None:
            x_out = x_out / self.num_neighbors**0.5

        return self.linear_output(x_out)

    def message(self, x_j, edge_attr, edge_scalars):
        # Generate weights from edge scalars
        if edge_scalars is not None:
            weight = self.fc(edge_scalars)
        else:
            weight = self.fc(edge_attr)

        # Tensor product
        return self.tp(x_j, edge_attr, weight)


class MagneticFieldGNN(nn.Module):
    """
    Full equivariant GNN for predicting magnetic fields from atomic structures.

    Input features:
    - Node type (scalar, one-hot): 2D
    - Spin moment (vector): 3D
    - Optional static features (scalar): variable dimension

    Output:
    - Magnetic field (vector): 3D
    """
    def __init__(
        self,
        num_node_types=2,
        static_features_dim=0,
        hidden_features=64,
        num_layers=4,
        max_radius=5.0,
        num_neighbors=20.0,
        lmax=2,
    ):
        super().__init__()

        self.num_node_types = num_node_types
        self.static_features_dim = static_features_dim
        self.lmax = lmax
        self.max_radius = max_radius

        # Calculate total scalar input dimension
        scalar_dim = num_node_types + static_features_dim

        # Define irreps for different stages
        # Input: scalars (node type + static) + vectors (spin moment)
        self.irreps_node_input = o3.Irreps(f"{scalar_dim}x0e + 1x1o")  # scalars + 1 vector

        # Hidden: scalars + vectors + optional higher order
        self.irreps_node_hidden = o3.Irreps(f"{hidden_features}x0e + {hidden_features//2}x1o + {hidden_features//4}x2e")

        # Edge attributes: spherical harmonics up to lmax
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)

        # Output: 1 vector (magnetic field)
        self.irreps_node_output = o3.Irreps("1x1o")

        # Embedding layer
        self.embedding = Linear(
            irreps_in=self.irreps_node_input,
            irreps_out=self.irreps_node_hidden
        )

        # Message passing layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = E3EquivariantMPNN(
                irreps_node_input=self.irreps_node_hidden,
                irreps_node_hidden=self.irreps_node_hidden,
                irreps_edge_attr=self.irreps_edge_attr,
                irreps_node_output=self.irreps_node_hidden if i < num_layers - 1 else self.irreps_node_hidden,
                num_neighbors=num_neighbors,
            )
            self.layers.append(layer)

        # Final output layer
        self.output_layer = Linear(
            irreps_in=self.irreps_node_hidden,
            irreps_out=self.irreps_node_output
        )

    def forward(self, data: Data):
        """
        Forward pass.

        Args:
            data: PyG Data object with:
                - x: node features [N, num_node_types + 3 + static_features_dim]
                     First num_node_types dimensions are one-hot node type
                     Next 3 dimensions are spin moment vector
                     Remaining dimensions are static features
                - edge_index: [2, E]
                - pos: atomic positions [N, 3]
                - edge_attr: optional precomputed edge features

        Returns:
            Predicted magnetic field vectors [N, 3]
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Split features into scalars and vectors
        scalar_end_idx = self.num_node_types + self.static_features_dim
        scalars = x[:, :scalar_end_idx]  # [N, scalar_dim]
        vectors = x[:, scalar_end_idx:scalar_end_idx+3]  # [N, 3]

        # Create equivariant input: concatenate scalars and vectors as irreps
        # Scalars: type 0e (even parity scalars)
        # Vectors: type 1o (odd parity vectors, like pseudovectors)
        x_equivariant = torch.cat([scalars, vectors], dim=1)  # [N, scalar_dim + 3]

        # Always compute edge attributes using the model's lmax to ensure consistency
        edge_attr = self.compute_edge_attr(edge_attr)

        # Embedding
        x_hidden = self.embedding(x_equivariant)

        # Message passing
        for layer in self.layers:
            x_hidden = layer(x_hidden, edge_index, edge_attr)
            # Optional: add residual connection
            # x_hidden = x_hidden + layer(x_hidden, edge_index, edge_attr)

        # Output projection
        out = self.output_layer(x_hidden)

        return out

    def compute_edge_attr(self, edge_attr):
        """
        Compute spherical harmonic edge attributes from positions.
        """

        # Compute distances
        distances = vectors = edge_attr[:, 1:4]  # [E, 1]

        # Normalize to get unit vectors
        unit_vectors = edge_attr[:, :3] # [E, 3]

        # Compute spherical harmonics
        edge_sh = o3.spherical_harmonics(
            l=list(range(self.lmax + 1)),
            x=unit_vectors,
            normalize=True,
            normalization='component'
        )

        return edge_sh
