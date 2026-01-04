import torch
import numpy as np
from scipy.special import sph_harm

from tqdm import tqdm
from torch.nn.functional import mse_loss

"""
Utils for FeGdSpinGNN: normalization, basis transformation, training and evaluation loops, dataset statistics.

Callable functions:
    compute_normalization_stats(dataset)
    normalize_data(data, stats)
    denormalize_targets(y_norm, stats)
    basis_transformation(coord, l_max=2)
    train_one_epoch(model, loader, optimizer, device)
    evaluate(model, loader, device)
    Statistics(dataset) class for dataset statistics.
"""

#################### NORMALIZATION UTILITIES ####################
def compute_normalization_stats(dataset):
    """
    Compute mean and std for moments, edge_attr, and targets from dataset.
    
    Args:
        dataset: List of PyG Data objects or FeGdMagneticDataset
        
    Returns:
        stats: Dict with normalization parameters
    """
    # Node features: [node_type (2), moment (3)]
    moments = torch.cat([data.x[:, 2:5] for data in dataset], dim=0)  # columns 2,3,4 are M_x, M_y, M_z
    # Edge attributes: [dx, dy, dz, rij]. ([dx, dy, dz] are unit vectors if preprocessed)
    edge_dist = torch.cat([data.edge_attr[:, 3:4] for data in dataset], dim=0) 
    targets = torch.cat([data.y for data in dataset], dim=0)
    
    stats = {
        'moment_mean': moments.mean(dim=0),
        'moment_std': moments.std(dim=0).clamp(min=1e-6),
        'edge_dist_mean': edge_dist.mean(dim=0),
        'edge_dist_std': edge_dist.std(dim=0).clamp(min=1e-6),
        'target_mean': targets.mean(dim=0),
        'target_std': targets.std(dim=0).clamp(min=1e-6)
    }  
    return stats


def normalize_data(data, stats):
    """
    Normalize moments, edge_dist, and targets in a graph.
    Node features: [node_type (2), moment (3)], only moments are normalized; node_type stays unchanged.
    
    Args:
        data: PyG Data object
        stats: Dict from compute_normalization_stats()
        
    Returns:
        Normalized Data object (modifies in place)
    """
    # Normalize moments (columns 2, 3, 4)
    data.x[:, 2:5] = (data.x[:, 2:5] - stats['moment_mean']) / stats['moment_std']
    
    # Normalize edges and targets
    data.edge_attr[:, 3:4] = (data.edge_attr[:, 3:4] - stats['edge_dist_mean']) / stats['edge_dist_std']
    data.y = (data.y - stats['target_mean']) / stats['target_std']
    
    return data


def denormalize_targets(y_norm, stats):
    """
    Convert normalized predictions back to original scale.
    
    Args:
        y_norm: Normalized predictions tensor
        stats: Dict from compute_normalization_stats()
        
    Returns:
        Denormalized predictions
    """
    return y_norm * stats['target_std'] + stats['target_mean']


##################### BASIS TRANSFORMATION UTILITIES ####################
def basis_transformation(coord, l_max=2):
    """
    Apply basis transformation to edge attributes into spherical harmonics basis to make rotationally equivariant. 
    Edge attributes of the angular momentum will be direction, magnitude and spherical harmonics.
    
    Args:
        data: Matrix of coordinates [xs, ys, zs] (num_data, 3)
        l_max: Int of highest order of harmonics
    Returns:
        Tensor of transformed coordinates in spherical harmonics basis.
    """

    # HELPER FUNCTIONS
    def cartesian_to_spherical(x, y, z):
        '''
        Converting cartesian coordinates to spherical coordinates

        Args:
            x, y, z vectors of coordniates.
        Returns: 
            Transformed spherical basis 
        '''
        r = np.sqrt(x**2 + y**2 + z**2) # magnitude?
        theta = np.arccos(z/r) # polar angle
        phi = np.arctan2(y,x) # azimuthal angle

        return r, theta, phi
    
    def all_sph_harm(theta, phi, l_max):
        """
        Get all spherical harmonics (Y_l_m) for all combinations of degrees (l) and order (m). 
        The higher l gives a more detailed harmonic, also more data points.
        Args:
            theta and phi: angles for angular momentum
            l_max: degree of the harmonic 
        Returns: 
            Array of spherical harmonics of the angular momentum.
        """
        results = []
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                Y_lm = sph_harm(m, l, theta, phi)
                results.append(Y_lm)
        return np.array(results)   # shape = ((l_max+1)^2,)

    xs, ys, zs = coord[:, 0], coord[:, 1], coord[:, 2] #x, y, z
    r, theta, phi = cartesian_to_spherical(xs, ys, zs)

    Y_lm = all_sph_harm(theta, phi, l_max)

    res = np.concatenate([r[:, np.newaxis], Y_lm.T], axis=1)  # shape = (num_edges, 1 + (l_max+1)^2)
    
    return res

##################### MODEL TRAINERS and VALIDATION  ####################


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)

        pred = model(batch)     # [num_nodes_total, 3]
        target = batch.y        # [num_nodes_total, 3]

        loss = mse_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            batch = batch.to(device)

            pred = model(batch)
            target = batch.y

            loss = mse_loss(pred, target)
            total_loss += float(loss.item())

    return total_loss / len(loader)

class Statistics:
    """
    Compute basic statistics of the dataset: number of each node, b-field of each node, 
    spin of each node, atom type of each node and more.

    Args:
        dataset: PyTorch Geometric Dataset (e.g., FeGdMagneticDataset)
    Callable methods:
        summary(): Print a summary of dataset statistics.
    Callable attributes:
        n_samples: Total number of samples in the dataset.
        n_nodes_mean, n_nodes_std, n_nodes_min, n_nodes_max: Statistics for number of nodes per graph. 
        n_edges_mean, n_edges_std, n_edges_min, n_edges_max: Statistics for number of edges per graph.
        fe_count_mean, fe_count_std: Mean and std of number of Fe atoms per graph.
        gd_count_mean, gd_count_std: Mean and std of number of Gd atoms per
        graph.
        moment_mag_mean, moment_mag_std, moment_mag_min, moment_mag_max: Statistics for
        spin moment magnitudes across all nodes.
        b_field_mag_mean, b_field_mag_std, b_field_mag_min, b_field_mag_max: Statistics for
        magnetic field magnitudes across all nodes.
    """
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.n_samples = len(dataset)
        self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute all statistics from the dataset."""
        # Initialize lists to collect data
        n_nodes_list = []
        n_edges_list = []
        fe_counts = []
        gd_counts = []
        moment_magnitudes = []
        b_field_magnitudes = []
        neighbor_counts = []
        
        # Iterate through dataset
        for data in tqdm(self.dataset, desc="Computing statistics"):
            n_nodes = data.x.shape[0]
            n_edges = data.edge_index.shape[1]
            n_nodes_list.append(n_nodes)
            n_edges_list.append(n_edges)
            
            # Count Fe and Gd atoms (node_type one-hot: Fe=[1,0], Gd=[0,1])
            fe_count = (data.x[:, 0] > 0.5).sum().item()
            gd_count = (data.x[:, 1] > 0.5).sum().item()
            fe_counts.append(fe_count)
            gd_counts.append(gd_count)
            
            # Compute spin moment magnitudes
            moments = data.x[:, 2:5]  # M_x, M_y, M_z
            moment_mag = torch.norm(moments, dim=1)
            moment_magnitudes.append(moment_mag)
            
            # Compute magnetic field magnitudes
            b_field_mag = torch.norm(data.y, dim=1)
            b_field_magnitudes.append(b_field_mag)
            
            # Compute number of neighbors per node (node degree)
            node_degrees = torch.bincount(data.edge_index[0], minlength=n_nodes).float()
            neighbor_counts.append(node_degrees)
        
        # Compute aggregate statistics
        self.n_nodes_mean = np.mean(n_nodes_list)
        self.n_nodes_std = np.std(n_nodes_list)
        self.n_nodes_min = np.min(n_nodes_list)
        self.n_nodes_max = np.max(n_nodes_list)
        
        self.n_edges_mean = np.mean(n_edges_list)
        self.n_edges_std = np.std(n_edges_list)
        self.n_edges_min = np.min(n_edges_list)
        self.n_edges_max = np.max(n_edges_list)
        
        self.fe_count_mean = np.mean(fe_counts)
        self.fe_count_std = np.std(fe_counts)
        self.gd_count_mean = np.mean(gd_counts)
        self.gd_count_std = np.std(gd_counts)
        
        # Moment statistics
        moment_mag_all = torch.cat(moment_magnitudes)
        self.moment_mag_mean = moment_mag_all.mean().item()
        self.moment_mag_std = moment_mag_all.std().item()
        self.moment_mag_min = moment_mag_all.min().item()
        self.moment_mag_max = moment_mag_all.max().item()
        
        # B-field statistics
        b_field_mag_all = torch.cat(b_field_magnitudes)
        self.b_field_mag_mean = b_field_mag_all.mean().item()
        self.b_field_mag_std = b_field_mag_all.std().item()
        self.b_field_mag_min = b_field_mag_all.min().item()
        self.b_field_mag_max = b_field_mag_all.max().item()
        
        # Neighbor count statistics
        neighbor_counts_all = torch.cat(neighbor_counts)
        self.neighbor_count_mean = neighbor_counts_all.mean().item()
        self.neighbor_count_std = neighbor_counts_all.std().item()
        self.neighbor_count_min = neighbor_counts_all.min().item()
        self.neighbor_count_max = neighbor_counts_all.max().item()
    def histograms(self, num_bins=50):
        """Generate histograms for various dataset attributes."""
        import matplotlib.pyplot as plt

        # Spin moment magnitudes
        moment_magnitudes = torch.cat([torch.norm(data.x[:, 2:5], dim=1) for data in self.dataset])
        plt.figure()
        plt.hist(moment_magnitudes.numpy(), bins=num_bins, alpha=0.7, color='red')
        plt.title('Histogram of Spin Moment Magnitudes')
        plt.xlabel('Moment Magnitude')
        plt.ylabel('Frequency')
        plt.show()

        # Spin moments components
        moment_x = torch.cat([data.x[:, 2] for data in self.dataset])
        moment_y = torch.cat([data.x[:, 3] for data in self.dataset])
        moment_z = torch.cat([data.x[:, 4] for data in self.dataset])
        plt.figure()
        plt.hist(moment_x.numpy(), bins=num_bins, alpha=0.5, label='M_x', color='blue')
        plt.hist(moment_y.numpy(), bins=num_bins, alpha=0.5, label='M_y', color='green')
        plt.hist(moment_z.numpy(), bins=num_bins, alpha=0.5, label='M_z', color='orange')
        plt.title('Histogram of Spin Moment Components')
        plt.xlabel('Moment Component Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()  

        # Magnetic field magnitudes
        b_field_magnitudes = torch.cat([torch.norm(data.y, dim=1) for data in self.dataset])
        plt.figure()
        plt.hist(b_field_magnitudes.numpy(), bins=num_bins, alpha=0.7, color='purple')
        plt.title('Histogram of Magnetic Field Magnitudes')
        plt.xlabel('B-field Magnitude')
        plt.ylabel('Frequency')
        plt.show()

        # Magnetic field components
        b_field_x = torch.cat([data.y[:, 0] for data in self.dataset])
        b_field_y = torch.cat([data.y[:, 1] for data in self.dataset])
        b_field_z = torch.cat([data.y[:, 2] for data in self.dataset])
        plt.figure()
        plt.hist(b_field_x.numpy(), bins=num_bins, alpha=0.5, label='B_x', color='cyan')
        plt.hist(b_field_y.numpy(), bins=num_bins, alpha=0.5, label='B_y', color='magenta')
        plt.hist(b_field_z.numpy(), bins=num_bins, alpha=0.5, label='B_z', color='yellow')
        plt.title('Histogram of Magnetic Field Components')
        plt.xlabel('B-field Component Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()


    def summary(self):
        """Print a summary of dataset statistics."""
        print("=" * 60)
        print("DATASET STATISTICS SUMMARY")
        print("=" * 60)
        print(f"\nNEIGHBORS PER NODE:")
        print(f"  Mean: {self.neighbor_count_mean:.2f} ± {self.neighbor_count_std:.2f}")
        print(f"  Range: [{self.neighbor_count_min:.0f}, {self.neighbor_count_max:.0f}]")
        print(f"\nSPIN MOMENT MAGNITUDE:")
        print(f"  Mean: {self.moment_mag_mean:.4f} ± {self.moment_mag_std:.4f}")
        print(f"  Range: [{self.moment_mag_min:.4f}, {self.moment_mag_max:.4f}]")
        print(f"\nMAGNETIC FIELD MAGNITUDE:")
        print(f"  Mean: {self.b_field_mag_mean:.4f} ± {self.b_field_mag_std:.4f}")
        print(f"  Range: [{self.b_field_mag_min:.4f}, {self.b_field_mag_max:.4f}]")
        print("=" * 60)

###################### Evaluation Utilities ######################
def evaluate_physical_metrics(model, loader, device, f, y_std):
    """Evaluate on original (denormalized) scale"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            
            # Denormalize predictions and targets
            out_denorm = out * y_std.to(device) + y_mean.to(device)
            y_denorm = batch.y * y_std.to(device) + y_mean.to(device)
            
            all_preds.append(out_denorm.cpu())
            all_targets.append(y_denorm.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    mse = torch.mean((all_preds - all_targets)**2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(all_preds - all_targets))
    
    # Per-component metrics
    mse_per_comp = torch.mean((all_preds - all_targets)**2, dim=0)
    
    return {
        'mse': mse.item(),
        'rmse': rmse.item(),
        'mae': mae.item(),
        'mse_x': mse_per_comp[0].item(),
        'mse_y': mse_per_comp[1].item(),
        'mse_z': mse_per_comp[2].item()
    }

###################### Evaluation Utilities ######################
def evaluate_physical_metrics(model, loader, device, y_mean, y_std):
    """Evaluate on original (denormalized) scale"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            
            # Denormalize predictions and targets
            out_denorm = out * y_std.to(device) + y_mean.to(device)
            y_denorm = batch.y * y_std.to(device) + y_mean.to(device)
            
            all_preds.append(out_denorm.cpu())
            all_targets.append(y_denorm.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    mse = torch.mean((all_preds - all_targets)**2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(all_preds - all_targets))
    
    # Per-component metrics
    mse_per_comp = torch.mean((all_preds - all_targets)**2, dim=0)
    
    return {
        'mse': mse.item(),
        'rmse': rmse.item(),
        'mae': mae.item(),
        'mse_x': mse_per_comp[0].item(),
        'mse_y': mse_per_comp[1].item(),
        'mse_z': mse_per_comp[2].item()
    }