import torch
import numpy as np
from scipy.special import sph_harm

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