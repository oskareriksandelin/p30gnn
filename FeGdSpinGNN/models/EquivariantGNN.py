import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from e3nn import o3
from e3nn.nn import Gate
import math


class EquivariantGNN(nn.Module):
    """
    E(3)-equivariant GNN using radial basis function weights
    with spherical field output head.
    """

    def __init__(
        self,
        hidden_irreps="64x0e + 32x1o + 16x2e",
        lmax=2,
        num_layers=4,
        max_radius=0.35,
        num_radial_basis=6,
        rmlp_dropout=0.0,
        conv_dropout=0.0,
        n_sphere_samples=256,
    ):
        super().__init__()

        self.lmax = lmax
        self.max_radius = max_radius
        self.n_sphere_samples = n_sphere_samples

        # input / hidden irreps
        self.node_input_irreps = o3.Irreps("3x0e + 1x1o")
        self.edge_sh_irreps = o3.Irreps.spherical_harmonics(lmax)
        self.hidden_irreps = o3.Irreps(hidden_irreps)


        # scalar field on the sphere
        self.output_irreps = o3.Irreps(
            o3.Irreps.spherical_harmonics(lmax)
        )

        self.input_layer = o3.Linear(self.node_input_irreps, self.hidden_irreps)

        self.conv_layers = nn.ModuleList([
            TPConvLayer(
                in_irreps=self.hidden_irreps,
                out_irreps=self.hidden_irreps,
                edge_sh_irreps=self.edge_sh_irreps,
                hidden_irreps=self.hidden_irreps,
                num_radial_basis=num_radial_basis,
                max_radius=max_radius,
                rmlp_dropout=rmlp_dropout,
                conv_dropout=conv_dropout
            )
            for _ in range(num_layers)
        ])

        # linear map -> spherical coefficients
        self.output_layer = o3.Linear(self.hidden_irreps, self.output_irreps)

        # fixed spherical grid
        dirs = torch.randn(n_sphere_samples, 3)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)

        Y = o3.spherical_harmonics(
            list(range(self.lmax + 1)),
            dirs,
            normalize=True,
            normalization="component"
        )

        self.register_buffer("sphere_dirs", dirs)      # [S, 3]
        self.register_buffer("sphere_Y", Y)            # [S, n_coeffs]

    def forward(self, data):
        # backbone
        x = self.input_layer(data.x)

        for conv in self.conv_layers:
            x = conv(
                x,
                data.edge_index,
                data.edge_attr,   # distances
                data.edge_sh
            )

        # spherical coefficients
        coeffs = self.output_layer(x) # shape: [N, n_coeffs]

        
        # evaluate scalar field on sphere
        # rho[n, s] = sum_c coeffs[n, c] * Y[s, c]
        rho = torch.einsum(
            "nc,sc->ns",
            coeffs,
            self.sphere_Y
        )

        # integrate to vector field
        B_pred = torch.einsum(
            "ns,sd->nd",
            rho,
            self.sphere_dirs
        ) / self.n_sphere_samples

        return B_pred


class TPConvLayer(MessagePassing):
    """
    E(3)-equivariant message passing with radial basis function weights.

    Key difference from basic approach:
    - Uses RBF embedding of distances to generate per-channel weights
    - Different message channels can have different distance dependencies
    - More expressive than single scalar distance feature
    """

    def __init__(
        self,
        in_irreps,
        out_irreps,
        edge_sh_irreps,  # spherical harmonics
        hidden_irreps,
        num_radial_basis=6,
        max_radius=0.35,
        rmlp_dropout=0.0,  # dropout for radial MLP
        conv_dropout=0.0   # dropout for convolution
    ):
        super().__init__(aggr='add', node_dim=0)

        self.in_irreps = o3.Irreps(in_irreps)
        self.out_irreps = o3.Irreps(out_irreps)
        self.edge_sh_irreps = o3.Irreps(edge_sh_irreps)
        self.conv_dropout = conv_dropout

        # radial basis embedding
        self.rbf = RadialBasisEmbedding(num_basis=num_radial_basis, cutoff=max_radius)

        
        # tensor product
        self.tp = o3.FullyConnectedTensorProduct(
            self.in_irreps,
            self.edge_sh_irreps,
            hidden_irreps,
            shared_weights=False
        )

        # get nr of paths in tensor product
        # each path corresponds to one (in_irrep, edge_irrep) -> out_irrep combination
        self.num_tp_weights = self.tp.weight_numel

        # MLP to generate distance-dependent weights for tensor product
        # Maps RBF features -> weights for each TP path
        mlp_hidden = min(32, self.num_tp_weights)  # Limit hidden size
        self.radial_mlp = nn.Sequential(
            nn.Linear(num_radial_basis, mlp_hidden),
            nn.SiLU(),
            nn.Dropout(rmlp_dropout),
            nn.Linear(mlp_hidden, self.num_tp_weights)
        )

        # gating setup
        scalars = o3.Irreps([(mul, ir) for mul, ir in hidden_irreps if ir.l == 0 and ir.p == 1])
        gated = o3.Irreps([(mul, ir) for mul, ir in hidden_irreps if ir.l > 0])

        if len(gated) > 0:
            gate_scalars = o3.Irreps([(mul, "0e") for mul, _ in gated])

            self.gate = Gate(
                irreps_scalars=scalars,
                act_scalars=[torch.nn.SiLU() for _ in scalars],
                irreps_gates=gate_scalars,
                act_gates=[torch.sigmoid for _ in gate_scalars],
                irreps_gated=gated
            )
            self.pre_gate = o3.Linear(hidden_irreps, self.gate.irreps_in)
        else:
            self.gate = None
            self.pre_gate = o3.Linear(hidden_irreps, hidden_irreps)

        # self-connection and output
        self.self_connection = o3.Linear(self.in_irreps, self.out_irreps)
        gate_out_irreps = self.gate.irreps_out if self.gate else hidden_irreps
        self.linear_out = o3.Linear(gate_out_irreps, self.out_irreps)

    def forward(self, x, edge_index, edge_distances, edge_sh):
        """
        Args:
            x: node features
            edge_index: edge connectivity
            edge_distances: [E, 1] distances (NOT in edge_attr anymore)
            edge_sh: [E, (lmax+1)^2] spherical harmonics only
        """
        out = self.self_connection(x)

        if self.conv_dropout > 0 and self.training:
            x = torch.nn.functional.dropout(x, p=self.conv_dropout, training=True)

        out = out + self.propagate(
            edge_index,
            x=x,
            edge_distances=edge_distances,
            edge_sh=edge_sh
        )
        return out

    def message(self, x_j, edge_distances, edge_sh):
        """
        Compute messages with distance-dependent weights.

        1. Embed distances using RBF -> [E, num_basis]
        2. MLP transforms RBF -> TP weights [E, num_tp_weights]
        3. Apply TP with these per-edge weights
        """
        # get RBF embedding of distances
        rbf = self.rbf(edge_distances)  # [E, num_basis]

        # generate per-edge weights for tensor product
        tp_weights = self.radial_mlp(rbf)  # [E, num_tp_weights]

        # apply tensor product with distance-dependent weights
        msg = self.tp(x_j, edge_sh, weight=tp_weights)

        # gating and output
        msg = self.pre_gate(msg)
        if self.gate:
            msg = self.gate(msg)
        msg = self.linear_out(msg)

        return msg
    

class RadialBasisEmbedding(nn.Module):
    """
    Radial basis function embedding for distances.

    Converts scalar distances into a higher-dimensional representation
    using Gaussian radial basis functions.

    Args:
        num_basis: number of basis functions
        cutoff: maximum distance
        learnable: whether basis centers and widths are learnable
    """
    def __init__(self, num_basis=6, cutoff=0.35, learnable=False):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff

        # Initialize Gaussian centers uniformly in [0, cutoff]
        centers = torch.linspace(0, cutoff, num_basis)
        if learnable:
            self.centers = nn.Parameter(centers)
        else:
            self.register_buffer('centers', centers)

        # Initialize widths based on spacing
        width = (cutoff / num_basis) * 0.5
        if learnable:
            self.widths = nn.Parameter(torch.ones(num_basis) * width)
        else:
            self.register_buffer('widths', torch.ones(num_basis) * width)

    def forward(self, distances):
        """
        Args:
            distances: [E, 1] edge distances
        Returns:
            rbf: [E, num_basis] radial basis function values
        """
        # distances: [E, 1], centers: [num_basis]
        # Compute Gaussian RBF: exp(-(d - c)^2 / (2*w^2))
        diff = distances - self.centers.view(1, -1)  # [E, num_basis]
        rbf = torch.exp(-0.5 * (diff / self.widths.view(1, -1))**2)

        # apply smooth cutoff
        cutoff_values = self._cosine_cutoff(distances)
        rbf = rbf * cutoff_values

        return rbf

    def _cosine_cutoff(self, distances):
        """Smooth cutoff function that goes to zero at self.cutoff"""
        cutoff_values = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoff_values = cutoff_values * (distances < self.cutoff).float()
        return cutoff_values