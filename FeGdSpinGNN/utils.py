import torch


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
    edge_attrs = torch.cat([data.edge_attr for data in dataset], dim=0)
    targets = torch.cat([data.y for data in dataset], dim=0)
    
    stats = {
        'moment_mean': moments.mean(dim=0),
        'moment_std': moments.std(dim=0).clamp(min=1e-6),
        'edge_mean': edge_attrs.mean(dim=0),
        'edge_std': edge_attrs.std(dim=0).clamp(min=1e-6),
        'target_mean': targets.mean(dim=0),
        'target_std': targets.std(dim=0).clamp(min=1e-6)
    }  
    return stats


def normalize_data(data, stats):
    """
    Normalize moments, edge_attr, and targets in a graph.
    Node features: [node_type (2), moment (3)]
    Only moments are normalized; node_type stays unchanged.
    
    Args:
        data: PyG Data object
        stats: Dict from compute_normalization_stats()
        
    Returns:
        Normalized Data object (modifies in place)
    """
    # Normalize moments (columns 2, 3, 4)
    data.x[:, 2:5] = (data.x[:, 2:5] - stats['moment_mean']) / stats['moment_std']
    
    # Normalize edges and targets
    data.edge_attr = (data.edge_attr - stats['edge_mean']) / stats['edge_std']
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